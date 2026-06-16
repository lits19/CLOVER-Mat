"""
src/rewards/phonon_reward.py
-----------------------------
PhononReward: reward based on phonon stability and vibrational properties.

Imaginary phonon modes indicate dynamical instability (the crystal will
spontaneously distort). A stable material has no imaginary frequencies.

Reward signals:
  1. Dynamical stability: no imaginary modes → binary reward
  2. Phonon band gap: gap between acoustic and optical branches
  3. Minimum phonon frequency: reward for absence of soft modes
  4. Debye temperature: estimated from phonon DOS

Backends:
  "mp_api"    – query Materials Project Phonon DB (phonopy results)
  "mace"      – compute harmonic force constants via MACE, check stability
  "chgnet"    – use CHGNet as force field for phonon calculation
  "heuristic" – use structure-based proxy for stability (tolerance factor, etc.)

Physical background:
  A harmonic phonon spectrum ω(k) is computed by diagonalizing the dynamical
  matrix D(q) = Σ Φ_αβ(l,l') exp(iq·R_l) / √(M_α M_β)
  Imaginary frequencies ↔ negative eigenvalues ↔ unstable crystal.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import torch
import numpy as np
from pymatgen.core import Structure

from src.rewards.base import RewardComponent


def _phonon_stability_via_mp_api(
    structures: List[Structure],
    api_key: Optional[str] = None,
) -> List[Optional[dict]]:
    """
    Query MP Phonon DB for phonon stability data.

    Returns list of dicts with keys:
      - "stable": bool (no imaginary modes)
      - "min_freq": float (minimum frequency in THz; negative = imaginary)
      - "debye_temp": float (K)
    """
    try:
        from mp_api.client import MPRester
    except ImportError:
        raise ImportError("pip install mp-api")

    key = api_key or os.environ.get("MP_API_KEY")
    if not key:
        raise ValueError("Set MP_API_KEY or pass api_key=")

    results: List[Optional[dict]] = []
    with MPRester(api_key=key) as mpr:
        for struct in structures:
            try:
                formula = struct.composition.reduced_formula
                docs = mpr.phonon.search(
                    formula=formula,
                    fields=["has_imaginary_modes", "min_freq", "debye_temperature"],
                )
                if docs:
                    d = docs[0]
                    results.append({
                        "stable": not getattr(d, "has_imaginary_modes", True),
                        "min_freq": getattr(d, "min_freq", None),
                        "debye_temp": getattr(d, "debye_temperature", None),
                    })
                else:
                    results.append(None)
            except Exception as e:
                warnings.warn(f"MP phonon query failed: {e}")
                results.append(None)
    return results


def _phonon_stability_via_mace(
    structures: List[Structure],
    model_path: str = "mace-mp-0b3-medium",
    device: str = "cpu",
    supercell_matrix: Optional[list] = None,
) -> List[Optional[dict]]:
    """
    Compute harmonic force constants using MACE and check phonon stability.

    Uses phonopy for the phonon calculation and MACE as the force engine.
    Requires: mace-torch, phonopy

    supercell_matrix: 3x3 list or None (defaults to [[2,0,0],[0,2,0],[0,0,2]])
    """
    try:
        from mace.calculators import mace_mp
        import phonopy
        from phonopy.interface.vasp import read_vasp_from_strings
        from phonopy import Phonopy
    except ImportError:
        raise ImportError(
            "Install required packages:\n"
            "  pip install mace-torch phonopy"
        )

    calc = mace_mp(model=model_path, device=device, default_dtype="float32")
    sc_matrix = supercell_matrix or [[2, 0, 0], [0, 2, 0], [0, 0, 2]]

    results: List[Optional[dict]] = []
    for struct in structures:
        try:
            from phonopy.structure.atoms import PhonopyAtoms
            cell = struct.lattice.matrix
            positions = struct.frac_coords
            numbers = [site.specie.number for site in struct.sites]
            masses = [site.specie.atomic_mass for site in struct.sites]

            phonopy_atoms = PhonopyAtoms(
                symbols=[site.specie.symbol for site in struct.sites],
                cell=cell,
                scaled_positions=positions,
                masses=masses,
            )

            ph = Phonopy(phonopy_atoms, supercell_matrix=sc_matrix)
            ph.generate_displacements(distance=0.01)

            # Compute forces for each displacement
            force_sets = []
            for sc in ph.supercells_with_displacements:
                ase_atoms = _phonopy_to_ase(sc)
                ase_atoms.calc = calc
                forces = ase_atoms.get_forces()
                force_sets.append(forces)

            ph.forces = force_sets
            ph.produce_force_constants()
            ph.run_mesh([10, 10, 10])
            ph.run_total_dos()

            # Check for imaginary modes
            mesh = ph.get_mesh_dict()
            freqs = mesh["frequencies"].flatten()  # THz
            min_freq = float(freqs.min())
            has_imaginary = bool(min_freq < -0.1)  # threshold: -0.1 THz

            # Debye temperature from DOS
            debye_temp = _debye_temp_from_dos(ph)

            results.append({
                "stable": not has_imaginary,
                "min_freq": min_freq,
                "debye_temp": debye_temp,
            })

        except Exception as e:
            warnings.warn(f"MACE phonon calculation failed: {e}")
            results.append(None)

    return results


def _phonopy_to_ase(phonopy_atoms):
    """Convert PhonopyAtoms to ASE Atoms."""
    try:
        from ase import Atoms
        return Atoms(
            symbols=phonopy_atoms.symbols,
            positions=phonopy_atoms.positions,
            cell=phonopy_atoms.cell,
            pbc=True,
        )
    except Exception as e:
        raise RuntimeError(f"PhonopyAtoms → ASE conversion failed: {e}")


def _debye_temp_from_dos(ph) -> Optional[float]:
    """Estimate Debye temperature from phonon DOS."""
    try:
        dos = ph.total_dos
        if dos is None:
            return None
        freqs = dos.frequency_points  # THz
        max_freq = float(freqs.max())
        # θ_D = ℏ·ω_max / k_B  (Debye cutoff approximation)
        HBAR_KB = 47.99  # K/THz  (ℏ/k_B in units of K·ps)
        return max_freq * HBAR_KB
    except Exception:
        return None


def _stability_heuristic(structure: Structure) -> dict:
    """
    Estimate phonon stability from structural descriptors.

    Uses empirical rules:
    - High symmetry structures tend to be more stable
    - Structures close to known stable phases (from composition) tend to be stable
    - Tolerance factor in perovskites correlates with stability
    """
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        sga = SpacegroupAnalyzer(structure, symprec=0.1)
        spacegroup_num = sga.get_space_group_number()
        # Higher symmetry (higher space group number not always better, but
        # cubic > hexagonal > orthorhombic in some heuristics)
        stability_score = min(1.0, spacegroup_num / 230)
    except Exception:
        stability_score = 0.5

    # Density check: too low density may indicate structural issues
    try:
        density = structure.density
        if density < 1.0 or density > 25.0:
            stability_score *= 0.5
    except Exception:
        pass

    return {
        "stable": stability_score > 0.3,
        "min_freq": stability_score * 5.0,  # proxy frequency in THz
        "debye_temp": stability_score * 400,  # proxy Debye temp in K
    }


@dataclass
class PhononReward(RewardComponent):
    """
    Reward based on phonon stability and vibrational properties.

    Parameters
    ----------
    reward_type : "stability" | "debye_temp" | "min_freq"
        What aspect of the phonon spectrum to reward:
        - "stability": +1 if no imaginary modes, -1 if unstable
        - "debye_temp": reward higher Debye temperature (stiffer lattice)
        - "min_freq": reward absence of soft modes (higher min frequency)
    backend : "mp_api" | "mace" | "heuristic"
    mace_model : str
        MACE model identifier (for "mace" backend).
    mace_device : str
    mp_api_key : str | None
    supercell_matrix : list | None
        3×3 supercell matrix for phonon calculation (MACE backend only).
    nan_penalty : float
    weight : float
    normalize : str
    """

    reward_type: Literal["stability", "debye_temp", "min_freq"] = "stability"
    backend: Literal["mp_api", "mace", "heuristic"] = "mp_api"
    mace_model: str = "mace-mp-0b3-medium"
    mace_device: str = "cpu"
    mp_api_key: Optional[str] = None
    supercell_matrix: Optional[list] = None
    nan_penalty: float = -1.0
    weight: float = 1.0
    normalize: str = "zscore"

    def compute(self, structures: List[Structure], **kwargs) -> torch.Tensor:
        if self.backend == "mp_api":
            phonon_data = _phonon_stability_via_mp_api(
                structures, api_key=self.mp_api_key
            )
        elif self.backend == "mace":
            phonon_data = _phonon_stability_via_mace(
                structures,
                model_path=self.mace_model,
                device=self.mace_device,
                supercell_matrix=self.supercell_matrix,
            )
        elif self.backend == "heuristic":
            phonon_data = [_stability_heuristic(s) for s in structures]
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        rewards = []
        for data in phonon_data:
            if data is None:
                rewards.append(self.nan_penalty)
            elif self.reward_type == "stability":
                rewards.append(1.0 if data["stable"] else -1.0)
            elif self.reward_type == "debye_temp":
                dt = data.get("debye_temp")
                if dt is None:
                    rewards.append(self.nan_penalty)
                else:
                    rewards.append(float(dt))
            elif self.reward_type == "min_freq":
                mf = data.get("min_freq")
                if mf is None:
                    rewards.append(self.nan_penalty)
                else:
                    rewards.append(float(mf))

        return torch.tensor(rewards, dtype=torch.float32)

    def stability_rate(self, structures: List[Structure]) -> float:
        """Fraction of dynamically stable structures."""
        if self.backend == "mp_api":
            data = _phonon_stability_via_mp_api(structures, api_key=self.mp_api_key)
        elif self.backend == "heuristic":
            data = [_stability_heuristic(s) for s in structures]
        else:
            data = [_stability_heuristic(s) for s in structures]

        stable = sum(1 for d in data if d is not None and d.get("stable", False))
        return stable / len(structures) if structures else 0.0
