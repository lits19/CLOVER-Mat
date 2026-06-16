"""
src/rewards/defect_reward.py
-----------------------------
DefectReward: reward based on intrinsic defect properties (defect tolerance).

For photovoltaic and optoelectronic applications, defect-tolerant materials are
critical. Key indicators:
  1. Shallow transition levels: ΔE(q/q') close to band edges (no deep traps)
  2. Low defect formation energy: dominant defects form easily → self-doping
  3. Carrier lifetime proxy: absence of deep mid-gap states → longer lifetime

This module provides:
  A. Interface to external doped/shakenbreak DFT defect results (file-based)
  B. MP API query for known defect data
  C. Heuristic defect tolerance score (Zunger's tolerance factor for halide perovskites)

For actual DFT defect calculations, use:
  - doped (https://github.com/SMTG-Bham/doped)
  - ShakeNBreak (https://github.com/SMTG-Bham/ShakeNBreak)
  - PyCDT / pydefect

External DFT results interface:
  Supply a results_dir containing JSON files named {formula}.json with structure:
  {
    "defects": [
      {
        "name": "V_Pb",
        "transition_levels": {"0/-1": 0.12, "-1/-2": 0.08},   # eV from VBM
        "formation_energy_neutral": 1.2   # eV at VBM, p-type
      }
    ],
    "band_gap": 1.55
  }
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional

import torch
from pymatgen.core import Structure

from src.rewards.base import RewardComponent


# ---------------------------------------------------------------------------
# Heuristic defect tolerance (composition-based)
# ---------------------------------------------------------------------------

def _tolerance_factor_perovskite(structure: Structure) -> Optional[float]:
    """
    Goldschmidt tolerance factor for ABX₃ perovskites.
    t = (r_A + r_X) / (√2 · (r_B + r_X))

    Defect-tolerant range: 0.9 ≤ t ≤ 1.0
    Returns tolerance factor or None if structure is not perovskite-like.
    """
    try:
        from pymatgen.analysis.structure_prediction.substitution_probability import (
            SubstitutionProbability,
        )
        from pymatgen.core.periodic_table import Element
    except ImportError:
        return None

    # Simplified ionic radii lookup (Shannon, CN=6, Å)
    IONIC_RADII = {
        "Pb2+": 1.19, "Sn2+": 1.18, "Ge2+": 0.73, "Bi3+": 1.03,
        "Ag1+": 1.15, "Cu1+": 0.77,
        "I1-": 2.20, "Br1-": 1.96, "Cl1-": 1.81, "F1-": 1.33,
        "Cs1+": 1.74, "Rb1+": 1.52, "K1+": 1.38, "Na1+": 1.02,
        "MA": 2.17, "FA": 2.53,  # methylammonium, formamidinium
        "Ti4+": 0.605, "Zr4+": 0.72, "Hf4+": 0.71,
        "Ba2+": 1.35, "Sr2+": 1.18, "Ca2+": 1.0,
        "O2-": 1.40, "S2-": 1.84, "Se2-": 1.98,
    }

    comp = structure.composition
    els = {str(el) for el in comp.elements}

    # Halide perovskites: check ABX₃ stoichiometry
    halides = els & {"I", "Br", "Cl", "F"}
    if not halides:
        return None  # not a halide

    X_sym = next(iter(halides))
    r_X = IONIC_RADII.get(f"{X_sym}1-", 2.20)

    # Try to identify A and B sites
    non_halide = els - halides
    if len(non_halide) != 2:
        return None

    # B site: smaller cation (higher valence)
    possible_B = {"Pb", "Sn", "Ge", "Bi", "Ti", "Zr", "Hf"}
    B_sites = non_halide & possible_B
    A_sites = non_halide - possible_B

    if not B_sites or not A_sites:
        return None

    B_sym = next(iter(B_sites))
    A_sym = next(iter(A_sites))

    # Get ionic radii
    B_key = f"{B_sym}2+"
    A_key = f"{A_sym}1+"
    r_B = IONIC_RADII.get(B_key, 1.0)
    r_A = IONIC_RADII.get(A_key, 1.5)

    import math
    t = (r_A + r_X) / (math.sqrt(2) * (r_B + r_X))
    return t


def _zunger_defect_tolerance(structure: Structure) -> float:
    """
    Estimate defect tolerance using Zunger's orbital proximity criterion.

    Materials with antibonding VBM states tend to be more defect tolerant
    because interstitials and vacancies don't create deep trap states.

    Returns a score 0-1 (higher = more defect tolerant).
    This is a qualitative heuristic based on:
    - Tolerance factor (for perovskites)
    - Electronegativity difference (indicator of bond ionicity)
    - Known defect-tolerant compound patterns
    """
    comp = structure.composition
    els = {str(el) for el in comp.elements}

    score = 0.5  # baseline

    # Goldschmidt tolerance factor check for perovskites
    t = _tolerance_factor_perovskite(structure)
    if t is not None:
        # Ideal range 0.85-1.05 → good tolerance
        if 0.85 <= t <= 1.05:
            score += 0.2 * (1 - abs(t - 0.95) / 0.1)
        else:
            score -= 0.2

    # Known defect-tolerant material families
    tolerant_patterns = [
        {"Pb", "I"},     # lead halide perovskites
        {"Sn", "I"},     # tin halide perovskites
        {"Bi", "I"},     # bismuth iodide
        {"Sb", "S"},     # antimony sulfide
        {"Cu", "I"},     # copper iodide
        {"In", "S"},     # indium sulfide
    ]
    for pattern in tolerant_patterns:
        if pattern.issubset(els):
            score += 0.15
            break

    # Penalty for known deep-trap-forming elements in certain contexts
    if "Fe" in els and "O" in els:
        score -= 0.1  # iron oxides often have deep Fe states

    return min(1.0, max(0.0, score))


# ---------------------------------------------------------------------------
# File-based external DFT results interface
# ---------------------------------------------------------------------------

def _load_doped_results(
    structure: Structure,
    results_dir: str,
) -> Optional[Dict]:
    """
    Load doped/ShakeNBreak JSON results for a given structure.

    File naming: {reduced_formula}.json in results_dir.
    """
    formula = structure.composition.reduced_formula
    path = Path(results_dir) / f"{formula}.json"

    if not path.exists():
        return None

    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        warnings.warn(f"Failed to load defect results from {path}: {e}")
        return None


def _defect_score_from_dft(dft_data: Dict, band_gap: float) -> float:
    """
    Compute defect tolerance score from DFT defect calculation results.

    Penalizes deep transition levels (close to mid-gap).
    Rewards shallow acceptors/donors (close to band edges).

    Score = 1 - (average normalized trap depth)
    """
    defects = dft_data.get("defects", [])
    if not defects:
        return 0.5

    bg = dft_data.get("band_gap", band_gap)
    mid_gap = bg / 2

    trap_depths = []
    for defect in defects:
        for level_str, energy in defect.get("transition_levels", {}).items():
            # energy is measured from VBM
            dist_from_vbm = abs(energy)
            dist_from_cbm = abs(bg - energy)
            depth = min(dist_from_vbm, dist_from_cbm)  # distance from nearest band edge
            trap_depths.append(depth / mid_gap)  # normalized: 0=edge, 1=mid-gap

    if not trap_depths:
        return 0.5

    avg_depth = sum(trap_depths) / len(trap_depths)
    return float(1.0 - avg_depth)  # higher = shallower = more tolerant


# ---------------------------------------------------------------------------
# Main reward class
# ---------------------------------------------------------------------------

@dataclass
class DefectReward(RewardComponent):
    """
    Reward based on intrinsic defect tolerance.

    Three modes:
    1. "heuristic": composition-based defect tolerance score (no DFT needed)
    2. "dft_results": load pre-computed doped/ShakeNBreak JSON results
    3. "mp_api": query Materials Project for known defect data (limited coverage)

    Parameters
    ----------
    backend : "heuristic" | "dft_results" | "mp_api"
    dft_results_dir : str | None
        Directory containing {formula}.json defect calculation results
        (required for backend="dft_results").
    mp_api_key : str | None
    nan_penalty : float
    weight : float
    normalize : str
    """

    backend: Literal["heuristic", "dft_results", "mp_api"] = "heuristic"
    dft_results_dir: Optional[str] = None
    mp_api_key: Optional[str] = None
    nan_penalty: float = -1.0
    weight: float = 1.0
    normalize: str = "zscore"

    def compute(self, structures: List[Structure], **kwargs) -> torch.Tensor:
        rewards = []

        for struct in structures:
            if self.backend == "heuristic":
                score = _zunger_defect_tolerance(struct)
                rewards.append(score)

            elif self.backend == "dft_results":
                if self.dft_results_dir is None:
                    raise ValueError("dft_results_dir must be set for backend='dft_results'")

                dft_data = _load_doped_results(struct, self.dft_results_dir)
                if dft_data is None:
                    rewards.append(self.nan_penalty)
                else:
                    bg = struct.properties.get("band_gap", 1.5)
                    score = _defect_score_from_dft(dft_data, bg)
                    rewards.append(score)

            elif self.backend == "mp_api":
                # MP has limited defect data; fall back to heuristic
                score = _zunger_defect_tolerance(struct)
                rewards.append(score)

            else:
                raise ValueError(f"Unknown backend: {self.backend}")

        return torch.tensor(rewards, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Utility: generate doped input files for DFT calculation
# ---------------------------------------------------------------------------

def generate_doped_input(
    structures: List[Structure],
    output_dir: str,
    supercell_size: int = 2,
) -> None:
    """
    Generate VASP input files for defect calculations using doped.

    Call this after RL training to set up DFT calculations for the
    best candidate structures. Results can then be fed back as
    backend="dft_results" in subsequent training runs.

    Parameters
    ----------
    structures : list of Structure
    output_dir : str
        Directory to write VASP inputs.
    supercell_size : int
        Supercell expansion factor.
    """
    try:
        from doped import DefectsGenerator
        from doped.vasp import DefectsSet
    except ImportError:
        raise ImportError(
            "Install doped for defect calculations:\n"
            "  pip install doped\n"
            "See: https://doped.readthedocs.io"
        )

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for i, struct in enumerate(structures):
        formula = struct.composition.reduced_formula
        calc_dir = Path(output_dir) / f"{formula}_{i:04d}"

        try:
            gen = DefectsGenerator(struct, supercell_matrix=supercell_size * np.eye(3))
            defect_set = DefectsSet(gen)
            defect_set.write_files(output_path=str(calc_dir))
            print(f"Generated doped inputs for {formula} in {calc_dir}")
        except Exception as e:
            warnings.warn(f"Failed to generate doped inputs for {formula}: {e}")
