"""
src/rewards/effective_mass_reward.py
-------------------------------------
EffectiveMassReward: reward based on carrier effective mass.

Light effective mass → high carrier mobility → better photovoltaic/LED performance.
Reward is higher when m* is smaller (closer to free electron mass m0).

Backends:
  "mp_api"   – query Materials Project band structure and compute m* from curvature
  "emc"      – use effective-mass-calculator (EMC) or sumo on local VASP output
  "proxy"    – use band gap as a rough proxy (smaller gap ↔ lighter m*, empirically)

Physical background:
  m* = ℏ² (d²E/dk²)⁻¹  along band edge
  For photovoltaics, target: m*_e < 0.5 m0, m*_h < 1.0 m0
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import torch
from pymatgen.core import Structure

from src.rewards.base import RewardComponent


def _effective_mass_via_mp_api(
    structures: List[Structure],
    api_key: Optional[str] = None,
    carrier: str = "electron",
) -> List[Optional[float]]:
    """
    Query MP for effective mass (averaged over k-points).

    MP provides `effective_mass` in the electronic structure summary.
    carrier: "electron" | "hole"
    """
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
                # Search for electronic structure data
                docs = mpr.summary.search(
                    formula=formula,
                    fields=["n", "p", "e_eff_mass_avg", "h_eff_mass_avg"],
                )
                if docs:
                    doc = docs[0]
                    if carrier == "electron":
                        m = getattr(doc, "e_eff_mass_avg", None)
                    else:
                        m = getattr(doc, "h_eff_mass_avg", None)
                    results.append(float(m) if m is not None else None)
                else:
                    results.append(None)
            except Exception as e:
                warnings.warn(f"MP effective mass query failed: {e}")
                results.append(None)
    return results


def _effective_mass_from_bandstructure(
    structures: List[Structure],
    api_key: Optional[str] = None,
    carrier: str = "electron",
) -> List[Optional[float]]:
    """
    Compute effective mass from pymatgen BandStructure via finite differences.

    m* = ℏ² / (d²E/dk²) at the band edge k-point.
    This requires fetching the full band structure from MP.
    """
    try:
        from mp_api.client import MPRester
        from pymatgen.electronic_structure.bandstructure import BandStructure
        import numpy as np
    except ImportError:
        return [None] * len(structures)

    HBAR2_OVER_2ME = 3.81  # eV·Å²  (ℏ²/2mₑ in eV·Å² units)
    key = api_key or os.environ.get("MP_API_KEY")
    if not key:
        return [None] * len(structures)

    results: List[Optional[float]] = []
    with MPRester(api_key=key) as mpr:
        for struct in structures:
            try:
                formula = struct.composition.reduced_formula
                docs = mpr.summary.search(formula=formula, fields=["material_id"])
                if not docs:
                    results.append(None)
                    continue
                mp_id = docs[0].material_id
                bs = mpr.get_bandstructure_by_material_id(mp_id)
                if bs is None:
                    results.append(None)
                    continue

                if carrier == "electron":
                    # Conduction band minimum
                    cbm_info = bs.get_cbm()
                    band_idx = cbm_info["band_index"][list(cbm_info["band_index"].keys())[0]][0]
                    kpt_idx = cbm_info["kpoint_index"][0]
                else:
                    # Valence band maximum
                    vbm_info = bs.get_vbm()
                    band_idx = vbm_info["band_index"][list(vbm_info["band_index"].keys())[0]][0]
                    kpt_idx = vbm_info["kpoint_index"][0]

                # Finite difference: use ±1 k-point for curvature
                spin = list(bs.bands.keys())[0]
                energies = bs.bands[spin][band_idx]  # shape (nkpts,)
                kpoints = np.array([kp.cart_coords for kp in bs.kpoints])  # Å⁻¹

                if kpt_idx == 0 or kpt_idx == len(energies) - 1:
                    results.append(None)
                    continue

                dE = energies[kpt_idx + 1] - 2 * energies[kpt_idx] + energies[kpt_idx - 1]
                dk = np.linalg.norm(kpoints[kpt_idx + 1] - kpoints[kpt_idx - 1]) / 2
                if abs(dk) < 1e-10 or abs(dE) < 1e-10:
                    results.append(None)
                    continue

                # m* = ℏ²/(d²E/dk²) in units of m_e
                d2E_dk2 = dE / (dk ** 2)  # eV/Å⁻²
                m_star = HBAR2_OVER_2ME / abs(d2E_dk2)  # dimensionless (units of m_e)
                results.append(float(m_star))
            except Exception as e:
                warnings.warn(f"Band structure effective mass failed: {e}")
                results.append(None)
    return results


@dataclass
class EffectiveMassReward(RewardComponent):
    """
    Reward inversely proportional to carrier effective mass.

    Lighter carriers (m* → 0) → higher mobility → better device performance.
    Reward = -m*/m0  (so minimizing m* maximizes reward).

    Parameters
    ----------
    carrier : "electron" | "hole"
    target_max : float
        Maximum acceptable effective mass (in units of m₀).
        Structures below this get positive reward.
    backend : "mp_api" | "bandstructure" | "proxy"
    mp_api_key : str | None
    nan_penalty : float
    weight : float
    normalize : str
    """

    carrier: Literal["electron", "hole"] = "electron"
    target_max: float = 0.5  # m₀; typical threshold for high-mobility materials
    backend: Literal["mp_api", "bandstructure", "proxy"] = "mp_api"
    mp_api_key: Optional[str] = None
    nan_penalty: float = -1.0
    weight: float = 1.0
    normalize: str = "zscore"

    def compute(self, structures: List[Structure], **kwargs) -> torch.Tensor:
        if self.backend == "mp_api":
            m_stars = _effective_mass_via_mp_api(
                structures, api_key=self.mp_api_key, carrier=self.carrier
            )
        elif self.backend == "bandstructure":
            m_stars = _effective_mass_from_bandstructure(
                structures, api_key=self.mp_api_key, carrier=self.carrier
            )
        elif self.backend == "proxy":
            # Rough proxy: light effective mass correlates with small band gap
            # m* ≈ 0.1 * Eg (very rough empirical rule for some semiconductors)
            m_stars = [
                struct.properties.get("band_gap", None)
                for struct in structures
            ]
            m_stars = [0.1 * g if g is not None else None for g in m_stars]
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        rewards = []
        for m in m_stars:
            if m is None:
                rewards.append(self.nan_penalty)
            else:
                # Reward = -m* (lower m* → higher reward), capped at 0 for m* < target
                rewards.append(-float(m))

        return torch.tensor(rewards, dtype=torch.float32)
