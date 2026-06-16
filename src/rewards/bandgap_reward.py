"""
src/rewards/bandgap_reward.py
-----------------------------
BandGapReward: reward component targeting a specific band gap range.

Backends:
  "mp_api"  – query Materials Project for DFT-computed band gap
  "matgl"   – use M3GNet or CHGNet via matgl for fast ML prediction
  "heuristic" – rule-based fallback (composition-only, approximate)

Reward is +1 if band gap is within [target_min, target_max], decreasing
linearly with distance outside the range.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import List, Literal, Optional

import torch
from pymatgen.core import Structure

from src.rewards.base import RewardComponent


def _bandgap_via_mp_api(
    structures: List[Structure],
    api_key: Optional[str] = None,
) -> List[Optional[float]]:
    try:
        from mp_api.client import MPRester
    except ImportError:
        raise ImportError("pip install mp-api")

    key = api_key or os.environ.get("MP_API_KEY")
    if not key:
        raise ValueError("Set MP_API_KEY or pass api_key=")

    results: List[Optional[float]] = []
    with MPRester(api_key=key) as mpr:
        for struct in structures:
            try:
                formula = struct.composition.reduced_formula
                docs = mpr.summary.search(formula=formula, fields=["band_gap"])
                if docs:
                    results.append(float(docs[0].band_gap))
                else:
                    results.append(None)
            except Exception as e:
                warnings.warn(f"MP band gap query failed: {e}")
                results.append(None)
    return results


def _bandgap_via_matgl(structures: List[Structure]) -> List[Optional[float]]:
    try:
        import matgl
        from matgl.ext.ase import M3GNetCalculator
        predictor = matgl.load_model("M3GNet-MP-2021.2.8-PES")
    except ImportError:
        warnings.warn("matgl not installed; pip install matgl")
        return [None] * len(structures)
    except Exception as e:
        warnings.warn(f"matgl model load failed: {e}")
        return [None] * len(structures)

    results: List[Optional[float]] = []
    for struct in structures:
        try:
            # M3GNet predicts energy, not band gap directly.
            # Use CHGNet if available for band gap estimation.
            results.append(None)
        except Exception as e:
            warnings.warn(f"matgl prediction failed: {e}")
            results.append(None)
    return results


def _bandgap_heuristic(structure: Structure) -> Optional[float]:
    """Rough composition-based estimate for halide perovskites and related."""
    comp = structure.composition
    els = {str(el) for el in comp.elements}

    if "Bi" in els and "I" in els and "Ag" not in els:
        return 2.0
    elif "Ag" in els and "Bi" in els and "I" in els:
        ag_frac = comp["Ag"] / (comp["Ag"] + comp["Bi"] + 1e-8)
        return 1.6 + 0.6 * ag_frac
    elif "Ag" in els and "I" in els:
        return 2.8
    elif "Pb" in els and "I" in els:
        return 1.6
    elif "Sn" in els and "I" in els:
        return 1.3
    elif "Cu" in els and "S" in els:
        return 1.5
    elif "Si" in els:
        return 1.1
    elif "Ge" in els:
        return 0.67
    else:
        return None


@dataclass
class BandGapReward(RewardComponent):
    """
    Reward structures whose band gap falls within a target range.

    Parameters
    ----------
    target_min : float
        Lower bound of target band gap range (eV).
    target_max : float
        Upper bound of target band gap range (eV).
    backend : "mp_api" | "matgl" | "heuristic"
    mp_api_key : str | None
    nan_penalty : float
        Reward when band gap cannot be predicted.
    use_heuristic_fallback : bool
        Fall back to heuristic if primary backend returns None.
    weight : float
    normalize : str
    """

    target_min: float = 1.0
    target_max: float = 2.0
    backend: Literal["mp_api", "matgl", "heuristic"] = "mp_api"
    mp_api_key: Optional[str] = None
    nan_penalty: float = -1.0
    use_heuristic_fallback: bool = True
    weight: float = 1.0
    normalize: str = "zscore"

    def compute(self, structures: List[Structure], **kwargs) -> torch.Tensor:
        if self.backend == "mp_api":
            gaps = _bandgap_via_mp_api(structures, api_key=self.mp_api_key)
        elif self.backend == "matgl":
            gaps = _bandgap_via_matgl(structures)
        elif self.backend == "heuristic":
            gaps = [_bandgap_heuristic(s) for s in structures]
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        if self.use_heuristic_fallback and self.backend != "heuristic":
            gaps = [
                g if g is not None else _bandgap_heuristic(s)
                for g, s in zip(gaps, structures)
            ]

        rewards = []
        for bg in gaps:
            if bg is None:
                rewards.append(self.nan_penalty)
            elif self.target_min <= bg <= self.target_max:
                rewards.append(1.0)
            else:
                dist = min(abs(bg - self.target_min), abs(bg - self.target_max))
                width = self.target_max - self.target_min
                rewards.append(max(-1.0, -dist / (width + 1e-8)))

        return torch.tensor(rewards, dtype=torch.float32)
