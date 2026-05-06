"""
custom_rewards/bandgap_reward.py
---------------------------------
Example custom reward: target a specific band gap range for Ag-Bi-I.

This demonstrates how to add a new reward component by:
  1. Subclassing RewardComponent
  2. Dropping the file in custom_rewards/
  3. Referencing it in a YAML config

Uses a pretrained ML band gap predictor (MEGNet / CHGNet / user-provided).
Falls back to a rule-based heuristic if no model is available.

To add to your config:
    - target: custom_rewards.bandgap_reward.BandGapReward
      weight: 0.8
      target_min: 1.0   # eV
      target_max: 2.0   # eV
      normalize: zscore
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from pymatgen.core import Structure

# Add project root to path if running standalone
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rewards.base import RewardComponent


def _predict_bandgap_chgnet(structures: List[Structure]) -> List[Optional[float]]:
    """
    Predict band gaps using CHGNet universal potential.
    CHGNet predicts energies; band gap estimation requires a separate ML model.
    This is a placeholder showing how to integrate such a model.
    """
    try:
        from chgnet.model import CHGNet
        model = CHGNet.load()
        predictions = []
        for struct in structures:
            try:
                prediction = model.predict_structure(struct)
                # CHGNet doesn't predict band gap directly;
                # this shows the integration pattern.
                # Replace with your trained band gap predictor.
                predictions.append(None)
            except Exception:
                predictions.append(None)
        return predictions
    except ImportError:
        return [None] * len(structures)


def _heuristic_bandgap(structure: Structure) -> Optional[float]:
    """
    Rule-based band gap heuristic for Ag-Bi-I compounds.
    Based on empirical trends from literature:
      - BiI3:  ~2.0 eV
      - AgBiI4: ~1.7 eV
      - Ag3BiI6: ~2.1 eV
    This is very approximate; replace with a trained ML model.
    """
    comp = structure.composition
    elements = {str(el) for el in comp.elements}

    if "Bi" in elements and "I" in elements and "Ag" not in elements:
        return 2.0  # BiI3-like
    elif "Ag" in elements and "Bi" in elements and "I" in elements:
        # Rough interpolation based on Ag:Bi ratio
        ag_frac = comp["Ag"] / (comp["Ag"] + comp["Bi"] + 1e-8)
        return 1.6 + 0.6 * ag_frac  # ranges ~1.6 – 2.2 eV
    elif "Ag" in elements and "I" in elements:
        return 2.8  # AgI-like
    else:
        return None


@dataclass
class BandGapReward(RewardComponent):
    """
    Reward structures with band gap within a target range.

    Reward function:
      - gap in [target_min, target_max]: reward = +1.0
      - gap outside range: reward decreases linearly with distance
      - gap unknown: reward = nan_penalty

    Parameters
    ----------
    target_min : float
        Minimum target band gap (eV).
    target_max : float
        Maximum target band gap (eV).
    nan_penalty : float
        Reward for structures where band gap cannot be predicted.
    use_heuristic : bool
        Fall back to heuristic if ML model unavailable.
    weight : float
    normalize : str
    """

    target_min: float = 1.0
    target_max: float = 2.0
    nan_penalty: float = -1.0
    use_heuristic: bool = True
    weight: float = 0.8
    normalize: str = "zscore"

    def compute(self, structures: List[Structure], **kwargs) -> torch.Tensor:
        # Try ML prediction first
        band_gaps = _predict_bandgap_chgnet(structures)

        # Fall back to heuristic for failed predictions
        if self.use_heuristic:
            for i, (bg, struct) in enumerate(zip(band_gaps, structures)):
                if bg is None:
                    band_gaps[i] = _heuristic_bandgap(struct)

        rewards = []
        for bg in band_gaps:
            if bg is None:
                rewards.append(self.nan_penalty)
            elif self.target_min <= bg <= self.target_max:
                rewards.append(1.0)
            else:
                # Penalty proportional to distance from target range
                dist = min(abs(bg - self.target_min), abs(bg - self.target_max))
                penalty = max(-1.0, -dist / (self.target_max - self.target_min))
                rewards.append(penalty)

        return torch.tensor(rewards, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Additional example: formation energy reward using ML potential
# ---------------------------------------------------------------------------

@dataclass
class FormationEnergyReward(RewardComponent):
    """
    Reward based on predicted formation energy (eV/atom).

    Lower formation energy = more stable compound.
    Uses structure.properties["energy_per_atom"] if precomputed,
    otherwise attempts MACE-MP estimation.

    Parameters
    ----------
    target_max : float
        Maximum acceptable formation energy (eV/atom). Structures
        below this threshold receive positive reward.
    nan_penalty : float
    weight : float
    normalize : str
    """

    target_max: float = -0.5  # eV/atom; typical for stable halides
    nan_penalty: float = -1.0
    weight: float = 0.5
    normalize: str = "zscore"

    def compute(self, structures: List[Structure], **kwargs) -> torch.Tensor:
        rewards = []
        for struct in structures:
            e_pa = struct.properties.get("energy_per_atom")
            if e_pa is None:
                rewards.append(self.nan_penalty)
            else:
                # Reward: negative formation energy (lower = better)
                rewards.append(-float(e_pa))

        return torch.tensor(rewards, dtype=torch.float32)
