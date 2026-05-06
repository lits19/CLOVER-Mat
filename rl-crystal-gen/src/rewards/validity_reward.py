"""
src/rewards/validity_reward.py
------------------------------
ValidityReward: checks charge neutrality and electronegativity balance
using the SMACT library (fast, no DFT required).

Also optionally checks structural validity:
  - Minimum inter-atomic distances (no overlapping atoms)
  - Reasonable density range
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from pymatgen.core import Structure

from src.rewards.base import RewardComponent


def _check_smact_validity(structure: Structure) -> Tuple[bool, bool]:
    """
    Returns (charge_neutral, electronegativity_ok).
    Falls back to (True, True) if smact is not installed.
    """
    try:
        import smact
        from smact.screening import smact_validity
    except ImportError:
        warnings.warn("smact not installed; skipping validity checks. pip install smact")
        return True, True

    composition = structure.composition
    species = []
    for el, amt in composition.items():
        try:
            smact_el = smact.Element(el.symbol)
            for ox in smact_el.oxidation_states:
                species.append((smact_el, ox, int(amt)))
        except Exception:
            continue

    if not species:
        return False, False

    # Use smact_validity for charge balance + electronegativity check
    try:
        cn_ok, en_ok = smact_validity(composition, threshold=1e-3)
        return cn_ok, en_ok
    except Exception:
        return False, False


def _check_min_distance(structure: Structure, min_dist: float = 0.5) -> bool:
    """Check that no two atoms are closer than min_dist Angstroms."""
    try:
        dm = structure.distance_matrix
        n = dm.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if dm[i, j] < min_dist:
                    return False
        return True
    except Exception:
        return False


def _check_density(
    structure: Structure,
    min_density: float = 0.5,
    max_density: float = 20.0,
) -> bool:
    """Check structure density is physically reasonable (g/cm³)."""
    try:
        d = structure.density
        return min_density <= d <= max_density
    except Exception:
        return False


@dataclass
class ValidityReward(RewardComponent):
    """
    Binary reward (0 or 1) for physically valid crystal structures.

    Checks applied (each can be toggled):
      1. SMACT charge neutrality
      2. SMACT electronegativity balance
      3. Minimum inter-atomic distance (no overlapping atoms)
      4. Density within physical range

    Parameters
    ----------
    check_charge : bool
        Enable charge neutrality check via SMACT.
    check_electronegativity : bool
        Enable electronegativity balance check via SMACT.
    check_min_dist : bool
        Enable minimum distance check.
    min_dist : float
        Minimum allowed inter-atomic distance in Angstroms.
    check_density : bool
        Enable density range check.
    min_density, max_density : float
        Allowed density range in g/cm³.
    weight : float
        Linear weight in composite reward.
    normalize : str
        Normalization; "none" is usually appropriate for binary rewards.
    """

    check_charge: bool = True
    check_electronegativity: bool = True
    check_min_dist: bool = True
    min_dist: float = 0.5
    check_density: bool = True
    min_density: float = 0.5
    max_density: float = 20.0
    weight: float = 0.5
    normalize: str = "none"

    def compute(self, structures: List[Structure], **kwargs) -> torch.Tensor:
        rewards = []
        for struct in structures:
            valid = True

            if self.check_charge or self.check_electronegativity:
                cn_ok, en_ok = _check_smact_validity(struct)
                if self.check_charge and not cn_ok:
                    valid = False
                if self.check_electronegativity and not en_ok:
                    valid = False

            if valid and self.check_min_dist:
                valid = _check_min_distance(struct, self.min_dist)

            if valid and self.check_density:
                valid = _check_density(struct, self.min_density, self.max_density)

            rewards.append(1.0 if valid else 0.0)

        return torch.tensor(rewards, dtype=torch.float32)

    def validity_rate(self, structures: List[Structure]) -> float:
        """Fraction of valid structures (0.0 – 1.0)."""
        return self.compute(structures).mean().item()
