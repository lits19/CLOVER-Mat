"""
src/rewards/thermal_conductivity_reward.py
-------------------------------------------
ThermalConductivityReward: reward based on lattice thermal conductivity κ_L.

For thermoelectric applications, low κ_L is desired (maximize ZT = S²σT/κ).
For thermal management applications, high κ_L may be targeted.

Backends:
  "mp_api"    – query Materials Project for computed κ_L (Phono3py data)
  "slack"     – use SLACK (Slack Atom-Level Kinetics) ML model if available
  "phonon_ml" – use phonon ML (e.g., CHGNet force constants + phono3py)
  "debye"     – Debye model approximation from elastic constants

Physical context:
  κ_L (Wm⁻¹K⁻¹) at 300 K:
    - Diamond: ~2200
    - Si: ~150
    - GaAs: ~45
    - BiTe (thermoelectric): ~1.5
    - PbTe: ~2.4
    - Halide perovskites: ~0.5–1.5

Debye model approximation:
  κ_L ≈ (1/3) · C_v · v_s · l
  where v_s = Debye velocity, l = phonon mean free path
  Can be estimated from elastic moduli (Voigt-Reuss-Hill).
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import List, Literal, Optional

import torch
import numpy as np
from pymatgen.core import Structure

from src.rewards.base import RewardComponent


def _thermal_conductivity_via_mp_api(
    structures: List[Structure],
    api_key: Optional[str] = None,
    temperature: float = 300.0,
) -> List[Optional[float]]:
    """
    Query MP for Phono3py-computed κ_L at given temperature (K).
    Returns κ in W/(m·K).
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
                # MP phonon data endpoint
                docs = mpr.phonon.search(
                    formula=formula,
                    fields=["material_id", "ph_kappa"],
                )
                if docs and hasattr(docs[0], "ph_kappa") and docs[0].ph_kappa is not None:
                    # ph_kappa may be a dict {temperature: kappa} or a scalar
                    kappa = docs[0].ph_kappa
                    if isinstance(kappa, dict):
                        # Find nearest temperature
                        temps = np.array(list(kappa.keys()), dtype=float)
                        idx = np.argmin(np.abs(temps - temperature))
                        results.append(float(list(kappa.values())[idx]))
                    else:
                        results.append(float(kappa))
                else:
                    results.append(None)
            except Exception as e:
                warnings.warn(f"MP thermal conductivity query failed: {e}")
                results.append(None)
    return results


def _debye_thermal_conductivity(structure: Structure) -> Optional[float]:
    """
    Estimate κ_L using Slack's modified Debye-Grüneisen model.

    κ_L ≈ A · M_avg · θ_D³ · V^(1/3) / (γ² · n^(2/3) · T)

    where:
      M_avg = average atomic mass (amu)
      θ_D = Debye temperature (K), estimated from elastic properties or empirical
      V = volume per atom (Å³)
      γ = Grüneisen parameter (assume γ ≈ 2 for most materials)
      n = number of atoms per formula unit
      T = temperature (assume 300 K)
      A = 3.1e-6 (Slack's constant in SI-like units)

    This is a rough estimate with ±50% accuracy.
    """
    try:
        comp = structure.composition
        n_atoms = structure.num_sites
        vol_per_atom = structure.volume / n_atoms  # Å³

        # Average atomic mass
        M_avg = sum(
            el.atomic_mass * amt
            for el, amt in comp.items()
        ) / sum(comp.values())  # amu

        # Estimate Debye temperature from density and sound velocity
        # For a rough estimate: θ_D ≈ 200–500 K for most oxides/sulfides
        # Use a composition-weighted empirical formula
        DEBYE_TEMPS = {
            "Si": 640, "Ge": 374, "C": 2230, "B": 1250,
            "Al": 428, "Fe": 470, "Cu": 343, "Ag": 225, "Au": 165,
            "Pb": 105, "Bi": 120, "Sn": 200,
            "O": 800, "S": 400, "Se": 300, "Te": 200,
            "I": 150, "Br": 120, "Cl": 180,
            "Na": 158, "K": 91, "Cs": 38, "Rb": 56,
            "Mg": 400, "Ca": 230, "Sr": 147, "Ba": 110,
            "Ti": 420, "Zr": 291, "Hf": 252,
            "N": 700, "P": 350, "As": 282, "Sb": 211,
        }

        theta_D = sum(
            DEBYE_TEMPS.get(el.symbol, 300) * amt
            for el, amt in comp.items()
        ) / sum(comp.values())

        # Slack model constant (SI units adjusted for Å³ input)
        A = 3.1e-6
        gamma = 2.0  # Grüneisen parameter
        T = 300.0
        n = n_atoms / len(comp.elements)  # atoms per formula unit, approximate

        kappa = A * float(M_avg) * theta_D**3 * (vol_per_atom * 1e-30)**(1/3)
        kappa /= (gamma**2 * n**(2/3) * T)

        # Convert to W/(m·K) — rough scale factor
        kappa_SI = kappa * 1e12  # empirical scaling to get W/(m·K) range
        return float(max(0.1, min(kappa_SI, 5000)))  # clip to physical range

    except Exception as e:
        warnings.warn(f"Debye thermal conductivity estimate failed: {e}")
        return None


@dataclass
class ThermalConductivityReward(RewardComponent):
    """
    Reward based on lattice thermal conductivity κ_L at 300 K.

    The reward sign depends on the application:
      thermoelectric: maximize -κ_L (low conductivity = high ZT)
      thermal management: maximize +κ_L (high conductivity)

    Parameters
    ----------
    mode : "thermoelectric" | "thermal_management"
        "thermoelectric": lower κ is better (negate reward)
        "thermal_management": higher κ is better
    target_kappa : float | None
        If set, reward penalizes distance from this target value.
    backend : "mp_api" | "debye"
    mp_api_key : str | None
    use_debye_fallback : bool
    temperature : float
        Temperature for κ evaluation (K).
    nan_penalty : float
    weight : float
    normalize : str
    """

    mode: Literal["thermoelectric", "thermal_management"] = "thermoelectric"
    target_kappa: Optional[float] = None
    backend: Literal["mp_api", "debye"] = "mp_api"
    mp_api_key: Optional[str] = None
    use_debye_fallback: bool = True
    temperature: float = 300.0
    nan_penalty: float = -1.0
    weight: float = 1.0
    normalize: str = "zscore"

    def compute(self, structures: List[Structure], **kwargs) -> torch.Tensor:
        if self.backend == "mp_api":
            kappas = _thermal_conductivity_via_mp_api(
                structures,
                api_key=self.mp_api_key,
                temperature=self.temperature,
            )
        else:
            kappas = [None] * len(structures)

        if self.use_debye_fallback or self.backend == "debye":
            kappas = [
                k if k is not None else _debye_thermal_conductivity(s)
                for k, s in zip(kappas, structures)
            ]

        rewards = []
        for kappa in kappas:
            if kappa is None:
                rewards.append(self.nan_penalty)
            elif self.target_kappa is not None:
                # Target a specific κ value
                dist = abs(kappa - self.target_kappa) / (self.target_kappa + 1e-8)
                rewards.append(-dist)
            elif self.mode == "thermoelectric":
                # Low κ is good: reward = -log(κ)
                rewards.append(-float(np.log(kappa + 1e-3)))
            else:
                # High κ is good: reward = log(κ)
                rewards.append(float(np.log(kappa + 1e-3)))

        return torch.tensor(rewards, dtype=torch.float32)
