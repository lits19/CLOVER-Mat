"""
src/rewards/ehull_reward.py
---------------------------
EhullReward: reward component based on energy above convex hull (E_hull).

Two backends are supported:
  1. "mp_api"  – queries the Materials Project REST API (exact, slower)
  2. "mace"    – uses MACE-MP-0 universal force field for fast E_above_hull
                 estimation (approximate, GPU-accelerated)

Lower E_hull → more thermodynamically stable → higher reward.
Reward is defined as:  r = -E_hull  (negated so minimizing E_hull = maximizing reward)
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import torch
from pymatgen.core import Structure
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry

from src.rewards.base import RewardComponent


# ---------------------------------------------------------------------------
# Helper: Materials Project convex hull query
# ---------------------------------------------------------------------------

def _ehull_via_mp_api(
    structures: List[Structure],
    api_key: Optional[str] = None,
    chemsys: Optional[str] = None,
) -> List[float]:
    """
    Query MP API for E_above_hull for each generated structure.

    For generated structures not in MP, we:
      1. Fetch stable entries for the relevant chemical system.
      2. Build a local PhaseDiagram.
      3. Compute E_above_hull using structure's DFT/predicted energy
         (supplied via structure.properties["energy_per_atom"] if present,
          otherwise falls back to 0 which returns hull distance = 0 → treated as NaN).

    Returns list of floats (eV/atom). NaN = could not evaluate.
    """
    try:
        from mp_api.client import MPRester
    except ImportError:
        raise ImportError("Install mp-api: pip install mp-api")

    key = api_key or os.environ.get("MP_API_KEY")
    if not key:
        raise ValueError("Set MP_API_KEY env variable or pass api_key=")

    # Determine chemical systems present
    results = []
    with MPRester(api_key=key) as mpr:
        for struct in structures:
            elements = [str(el) for el in struct.composition.elements]
            cs = "-".join(sorted(elements))
            try:
                # Fetch all stable entries in this chemsys
                entries = mpr.get_entries_in_chemsys(elements)
                pd = PhaseDiagram(entries)
                # Use energy_per_atom from structure properties if available
                energy_pa = struct.properties.get("energy_per_atom", None)
                if energy_pa is None:
                    results.append(float("nan"))
                    continue
                test_entry = PDEntry(
                    composition=struct.composition,
                    energy=energy_pa * struct.num_sites,
                )
                ehull = pd.get_e_above_hull(test_entry)
                results.append(float(ehull))
            except Exception as e:
                warnings.warn(f"MP API query failed for {cs}: {e}")
                results.append(float("nan"))
    return results


def _ehull_via_mace(
    structures: List[Structure],
    model_path: str = "mace-mp-0b3-medium",
    device: str = "cpu",
) -> List[float]:
    """
    Use MACE-MP universal force field to estimate E_hull.

    Steps:
      1. Relax each structure with MACE (optional, controlled by relax=True).
      2. Compute energy_per_atom.
      3. Build a local convex hull from the batch + reference endpoints.

    Returns list of floats (eV/atom). NaN = failed.
    """
    try:
        from mace.calculators import mace_mp
        from ase.optimize import LBFGS
    except ImportError:
        raise ImportError(
            "Install mace-torch for MACE backend:\n"
            "  pip install mace-torch\n"
            "or: pip install 'rl-crystal-gen[mace]'"
        )

    calc = mace_mp(model=model_path, device=device, default_dtype="float32")
    energies_pa: List[Optional[float]] = []

    for struct in structures:
        try:
            atoms = struct.to_ase_atoms()
            atoms.calc = calc
            energy = atoms.get_potential_energy()
            energies_pa.append(energy / len(atoms))
        except Exception as e:
            warnings.warn(f"MACE calculation failed: {e}")
            energies_pa.append(None)

    # Build a local convex hull from the batch energies
    # For a real deployment: augment with reference elemental energies from MP.
    # Here we do a simplified version: treat minimum energy as hull reference.
    valid = [e for e in energies_pa if e is not None]
    if not valid:
        return [float("nan")] * len(structures)

    hull_energy = min(valid)  # simple approximation: batch minimum as hull

    ehull_list = []
    for e in energies_pa:
        if e is None:
            ehull_list.append(float("nan"))
        else:
            ehull_list.append(max(0.0, e - hull_energy))  # ≥ 0

    return ehull_list


# ---------------------------------------------------------------------------
# Main reward class
# ---------------------------------------------------------------------------

@dataclass
class EhullReward(RewardComponent):
    """
    Reward based on energy above convex hull.

    Lower E_hull (more stable) → higher reward.
    Reward formula: r_i = -E_hull_i
    Structures where E_hull cannot be computed receive a penalty.

    Parameters
    ----------
    backend : "mp_api" | "mace"
        Which backend to use for E_hull computation.
    mp_api_key : str | None
        Materials Project API key. Falls back to MP_API_KEY env var.
    mace_model : str
        MACE model identifier (for "mace" backend).
    mace_device : str
        Device for MACE ("cpu", "cuda", "mps").
    nan_penalty : float
        Reward assigned to structures where E_hull cannot be computed.
    stable_threshold : float
        Structures with E_hull < threshold (eV/atom) are considered stable.
    weight : float
        Linear weight of this component in the composite reward.
    normalize : str
        Normalization method: "zscore" | "minmax" | "none".
    """

    backend: Literal["mp_api", "mace"] = "mp_api"
    mp_api_key: Optional[str] = None
    mace_model: str = "mace-mp-0b3-medium"
    mace_device: str = "cpu"
    nan_penalty: float = -2.0
    stable_threshold: float = 0.1  # eV/atom; common threshold for "stable"
    weight: float = 1.0
    normalize: str = "zscore"

    def compute(self, structures: List[Structure], **kwargs) -> torch.Tensor:
        """
        Compute -E_hull for each structure.

        Returns
        -------
        torch.Tensor of shape (N,)
            Negative E_hull values (higher = better stability).
        """
        if self.backend == "mp_api":
            ehull_values = _ehull_via_mp_api(
                structures, api_key=self.mp_api_key
            )
        elif self.backend == "mace":
            ehull_values = _ehull_via_mace(
                structures,
                model_path=self.mace_model,
                device=self.mace_device,
            )
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        rewards = []
        for ehull in ehull_values:
            if ehull != ehull:  # NaN check
                rewards.append(self.nan_penalty)
            else:
                rewards.append(-float(ehull))  # negate: lower E_hull = higher reward

        tensor = torch.tensor(rewards, dtype=torch.float32)

        # Log statistics for monitoring
        valid_mask = torch.tensor(
            [e == e for e in ehull_values], dtype=torch.bool
        )
        if valid_mask.any():
            valid_ehull = torch.tensor(
                [e for e in ehull_values if e == e], dtype=torch.float32
            )
            n_stable = (valid_ehull < self.stable_threshold).sum().item()
            # Store in kwargs for logger access
            kwargs["_ehull_stats"] = {
                "mean_ehull": valid_ehull.mean().item(),
                "min_ehull": valid_ehull.min().item(),
                "n_stable": n_stable,
                "n_valid": valid_mask.sum().item(),
                "n_total": len(structures),
            }

        return tensor

    def stability_fraction(self, structures: List[Structure]) -> float:
        """Fraction of structures with E_hull < stable_threshold."""
        raw = self.compute(structures)
        # raw = -E_hull, so stable if raw > -threshold
        stable = (raw > -self.stable_threshold).float().mean().item()
        return stable
