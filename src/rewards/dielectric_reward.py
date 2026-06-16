"""
src/rewards/dielectric_reward.py
---------------------------------
DielectricReward: reward based on static dielectric constant ε∞ and ε₀.

For optoelectronics:
  - High ε∞ (electronic, optical):  good for charge screening, exciton dissociation
  - Moderate ε₀ (static, total):    too high may indicate soft phonon modes / instability
  - Low exciton binding energy ∝ 1/ε∞² (Wannier-Mott model)

For photovoltaics: ε∞ > 5 is generally desirable.
For ferroelectrics: very high ε₀ is targeted.

Backends:
  "mp_api" – query Materials Project for computed dielectric tensors
  "proxy"  – use composition-based estimate via Clausius-Mossotti

Physical note:
  ε∞ = n² (n = refractive index at optical frequencies)
  ε₀ = ε∞ + ionic contribution (phonon-mediated)
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import torch
import numpy as np
from pymatgen.core import Structure

from src.rewards.base import RewardComponent


def _dielectric_via_mp_api(
    structures: List[Structure],
    api_key: Optional[str] = None,
) -> List[Optional[Tuple[float, float]]]:
    """
    Returns list of (e_electronic, e_ionic) tuples. None if not available.
    e_total = e_electronic + e_ionic.
    """
    try:
        from mp_api.client import MPRester
    except ImportError:
        raise ImportError("pip install mp-api")

    key = api_key or os.environ.get("MP_API_KEY")
    if not key:
        raise ValueError("Set MP_API_KEY or pass api_key=")

    results: List[Optional[Tuple[float, float]]] = []
    with MPRester(api_key=key) as mpr:
        for struct in structures:
            try:
                formula = struct.composition.reduced_formula
                docs = mpr.dielectric.search(
                    formula=formula,
                    fields=["e_electronic", "e_ionic"],
                )
                if docs:
                    d = docs[0]
                    e_elec = getattr(d, "e_electronic", None)
                    e_ion = getattr(d, "e_ionic", None)
                    # Convert tensor to scalar (isotropic average of diagonal)
                    if e_elec is not None and hasattr(e_elec, "__len__"):
                        e_elec = float(np.trace(np.array(e_elec)) / 3)
                    if e_ion is not None and hasattr(e_ion, "__len__"):
                        e_ion = float(np.trace(np.array(e_ion)) / 3)
                    if e_elec is not None and e_ion is not None:
                        results.append((float(e_elec), float(e_ion)))
                    else:
                        results.append(None)
                else:
                    results.append(None)
            except Exception as e:
                warnings.warn(f"MP dielectric query failed: {e}")
                results.append(None)
    return results


def _dielectric_clausius_mossotti(structure: Structure) -> Optional[float]:
    """
    Estimate electronic dielectric constant via Clausius-Mossotti relation.

    ε∞ = (1 + 2P) / (1 - P),  P = N·α / (3ε₀V)

    Uses atomic polarizabilities from a lookup table.
    This is a rough estimate; accuracy ±20–50%.
    """
    # Atomic polarizabilities in Å³ (from various sources)
    POLARIZABILITY = {
        "H": 0.667, "Li": 24.3, "Na": 23.6, "K": 43.4,
        "Mg": 10.6, "Ca": 22.8, "Sr": 27.6, "Ba": 39.7,
        "B": 3.03, "Al": 6.8, "Ga": 8.12, "In": 10.2,
        "Si": 5.38, "Ge": 6.07, "Sn": 7.7, "Pb": 6.8,
        "N": 1.1, "P": 3.63, "As": 4.31, "Sb": 6.6, "Bi": 7.4,
        "O": 0.802, "S": 2.9, "Se": 3.77, "Te": 5.5,
        "F": 0.557, "Cl": 2.18, "Br": 3.05, "I": 4.7,
        "Cu": 6.1, "Ag": 7.2, "Au": 5.8,
        "Zn": 5.75, "Cd": 7.2, "Hg": 5.02,
        "Ti": 14.0, "Zr": 17.9, "Hf": 16.2,
        "V": 12.4, "Nb": 15.7, "Ta": 13.1,
        "Mn": 9.4, "Fe": 8.4, "Co": 7.5, "Ni": 6.8,
    }

    try:
        comp = structure.composition
        vol = structure.volume  # Å³
        total_pol = 0.0
        for el, amt in comp.items():
            pol = POLARIZABILITY.get(el.symbol, 5.0)  # default 5.0 Å³
            total_pol += amt * pol

        # Clausius-Mossotti: P = total_pol / (3 * vol) * (4π/3) in CGS
        # In SI-like units with α in Å³ and V in Å³:
        # P = (4π/3) * N*α / V  →  ε = (1+2P)/(1-P)
        P = (4 * np.pi / 3) * total_pol / vol
        if P >= 1.0:
            return None  # unphysical
        eps = (1 + 2 * P) / (1 - P)
        return float(eps)
    except Exception:
        return None


@dataclass
class DielectricReward(RewardComponent):
    """
    Reward based on electronic (optical) or static dielectric constant.

    Parameters
    ----------
    component : "electronic" | "ionic" | "total"
        Which dielectric component to target.
    target_min : float
        Minimum acceptable dielectric constant.
        For solar cells: e_electronic > 5 (screens excitons).
    target_max : float | None
        Maximum acceptable value. None = no upper bound.
    backend : "mp_api" | "proxy"
    mp_api_key : str | None
    use_proxy_fallback : bool
        If True, fall back to Clausius-Mossotti when MP returns None.
    nan_penalty : float
    weight : float
    normalize : str
    """

    component: Literal["electronic", "ionic", "total"] = "electronic"
    target_min: float = 5.0
    target_max: Optional[float] = None
    backend: Literal["mp_api", "proxy"] = "mp_api"
    mp_api_key: Optional[str] = None
    use_proxy_fallback: bool = True
    nan_penalty: float = -1.0
    weight: float = 1.0
    normalize: str = "zscore"

    def compute(self, structures: List[Structure], **kwargs) -> torch.Tensor:
        if self.backend == "mp_api":
            raw = _dielectric_via_mp_api(structures, api_key=self.mp_api_key)
        else:
            raw = [None] * len(structures)

        rewards = []
        for i, (val, struct) in enumerate(zip(raw, structures)):
            eps = None
            if val is not None:
                e_elec, e_ion = val
                if self.component == "electronic":
                    eps = e_elec
                elif self.component == "ionic":
                    eps = e_ion
                else:
                    eps = e_elec + e_ion

            if eps is None and self.use_proxy_fallback:
                eps = _dielectric_clausius_mossotti(struct)

            if eps is None:
                rewards.append(self.nan_penalty)
            else:
                # Reward: positive if above target_min, penalize if below
                if eps >= self.target_min:
                    if self.target_max is None or eps <= self.target_max:
                        rewards.append(float(eps))
                    else:
                        # Above upper bound: soft penalty
                        excess = eps - self.target_max
                        rewards.append(float(self.target_max) - excess * 0.1)
                else:
                    # Below lower bound: penalty proportional to deficit
                    deficit = self.target_min - eps
                    rewards.append(-deficit)

        return torch.tensor(rewards, dtype=torch.float32)
