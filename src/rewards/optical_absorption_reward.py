"""
src/rewards/optical_absorption_reward.py
-----------------------------------------
OpticalAbsorptionReward: reward based on optical absorption coefficient α(ω).

For solar cell applications, a strong absorber should have:
  - High integrated absorption in the visible range (1.77–3.1 eV, 400–700 nm)
  - Absorption onset near the band gap (direct transitions preferred)

Backends:
  "mp_api"     – fetch dielectric function ε(ω) from MP, compute α(ω) = (ω/nc) Im[ε(ω)]
  "slme"       – compute Spectroscopic Limited Maximum Efficiency (SLME) as a proxy
  "dielectric" – use static dielectric from MP as a proxy for optical response

Physical formula:
  α(ω) = (√2 · ω/c) · √(√(ε₁²+ε₂²) - ε₁)   [SI units; converted to cm⁻¹]
  Integrated absorption: A = ∫ α(E) · AM1.5(E) dE over visible range
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

# AM1.5 solar photon flux tabulated at common energies (eV): rough values
# Full table should come from NREL; these are approximate for fast use.
_AM15_ENERGIES_EV = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2])
# Photon flux (photons/m²/s/eV), approximate AM1.5G values
_AM15_FLUX = np.array([3.0e21, 2.8e21, 2.5e21, 2.2e21, 1.8e21, 1.5e21,
                        1.2e21, 9.0e20, 7.0e20, 5.0e20, 3.5e20, 2.0e20])


def _absorption_from_dielectric(
    eps1: np.ndarray,
    eps2: np.ndarray,
    energies_ev: np.ndarray,
) -> np.ndarray:
    """
    Compute α(E) in cm⁻¹ from real (ε₁) and imaginary (ε₂) dielectric function.

    α(E) = (√2 · E/ℏc) · √(√(ε₁²+ε₂²) - ε₁)
    With E in eV and output in cm⁻¹:
      prefactor = E[eV] / (ℏc) = E[eV] / (1.97e-5 eV·cm) = E[eV] * 5.07e4 cm⁻¹/eV
    """
    eps_mod = np.sqrt(eps1**2 + eps2**2)
    k = np.sqrt(np.maximum((eps_mod - eps1) / 2, 0))  # extinction coefficient
    HBAR_C_EV_CM = 1.9732705e-5  # eV·cm
    alpha = 2 * energies_ev * k / HBAR_C_EV_CM  # cm⁻¹
    return alpha


def _absorption_via_mp_api(
    structures: List[Structure],
    api_key: Optional[str] = None,
    energy_min: float = 1.0,
    energy_max: float = 3.5,
) -> List[Optional[float]]:
    """
    Fetch optical dielectric function from MP and compute integrated absorption.
    Returns integrated α·φ_AM1.5 (relative units, higher = better absorber).
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
                docs = mpr.summary.search(formula=formula, fields=["material_id"])
                if not docs:
                    results.append(None)
                    continue

                mp_id = docs[0].material_id
                # Fetch optical properties (dielectric function)
                try:
                    opt = mpr.dielectric.get_data_by_id(mp_id)
                except Exception:
                    results.append(None)
                    continue

                if opt is None:
                    results.append(None)
                    continue

                # opt.epsilon_imaginary: list of [freq, eps2_xx, eps2_yy, eps2_zz]
                # Use isotropic average
                freq = np.array([row[0] for row in opt.epsilon_imaginary])  # eV
                eps2 = np.array([
                    (row[1] + row[2] + row[3]) / 3
                    for row in opt.epsilon_imaginary
                ])
                eps1_data = opt.epsilon_real
                eps1 = np.array([
                    (row[1] + row[2] + row[3]) / 3
                    for row in eps1_data
                ])

                # Compute absorption coefficient
                alpha = _absorption_from_dielectric(eps1, eps2, freq)

                # Integrate over AM1.5 visible range
                mask = (freq >= energy_min) & (freq <= energy_max)
                if not mask.any():
                    results.append(0.0)
                    continue

                flux_interp = np.interp(freq[mask], _AM15_ENERGIES_EV, _AM15_FLUX)
                integral = np.trapz(alpha[mask] * flux_interp, freq[mask])
                results.append(float(integral))

            except Exception as e:
                warnings.warn(f"Optical absorption query failed: {e}")
                results.append(None)
    return results


def _slme_proxy(
    structures: List[Structure],
    api_key: Optional[str] = None,
) -> List[Optional[float]]:
    """
    Compute Spectroscopic Limited Maximum Efficiency (SLME) using pymatgen.

    SLME accounts for band gap, direct/indirect nature, and spectral mismatch.
    Higher SLME → better single-junction solar cell absorber.
    """
    try:
        from mp_api.client import MPRester
        from pymatgen.analysis.solar import slme
    except ImportError:
        warnings.warn("mp-api or pymatgen.analysis.solar not available")
        return [None] * len(structures)

    key = api_key or os.environ.get("MP_API_KEY")
    if not key:
        return [None] * len(structures)

    results: List[Optional[float]] = []
    with MPRester(api_key=key) as mpr:
        for struct in structures:
            try:
                formula = struct.composition.reduced_formula
                docs = mpr.summary.search(formula=formula, fields=["material_id", "band_gap"])
                if not docs:
                    results.append(None)
                    continue

                doc = docs[0]
                bg = getattr(doc, "band_gap", None)
                if bg is None or bg <= 0:
                    results.append(None)
                    continue

                mp_id = doc.material_id
                try:
                    bs = mpr.get_bandstructure_by_material_id(mp_id)
                    if bs is not None:
                        slme_val = slme(bs, T=300)
                        results.append(float(slme_val) if slme_val is not None else None)
                    else:
                        results.append(None)
                except Exception:
                    results.append(None)
            except Exception as e:
                warnings.warn(f"SLME computation failed: {e}")
                results.append(None)
    return results


@dataclass
class OpticalAbsorptionReward(RewardComponent):
    """
    Reward based on optical absorption strength in the solar spectrum.

    Targets materials with high absorption coefficient in the visible range,
    suitable for thin-film photovoltaic absorbers.

    Parameters
    ----------
    backend : "mp_api" | "slme"
        "mp_api": compute integrated α(ω)·φ_AM1.5 from dielectric function
        "slme": compute Spectroscopic Limited Maximum Efficiency
    energy_min, energy_max : float
        Integration range in eV (default: visible range 1.0–3.5 eV).
    mp_api_key : str | None
    nan_penalty : float
    weight : float
    normalize : str
    """

    backend: Literal["mp_api", "slme"] = "mp_api"
    energy_min: float = 1.0   # eV
    energy_max: float = 3.5   # eV
    mp_api_key: Optional[str] = None
    nan_penalty: float = -1.0
    weight: float = 1.0
    normalize: str = "zscore"

    def compute(self, structures: List[Structure], **kwargs) -> torch.Tensor:
        if self.backend == "mp_api":
            scores = _absorption_via_mp_api(
                structures,
                api_key=self.mp_api_key,
                energy_min=self.energy_min,
                energy_max=self.energy_max,
            )
        elif self.backend == "slme":
            scores = _slme_proxy(structures, api_key=self.mp_api_key)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        rewards = [
            float(s) if s is not None else self.nan_penalty
            for s in scores
        ]
        return torch.tensor(rewards, dtype=torch.float32)
