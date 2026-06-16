"""
src/rewards/__init__.py
Reward components for RL-guided crystal structure generation.
"""

from src.rewards.base import RewardComponent, CompositeReward
from src.rewards.ehull_reward import EhullReward
from src.rewards.validity_reward import ValidityReward
from src.rewards.diversity_reward import DiversityReward, NoveltyReward
from src.rewards.composition_reward import CompositionReward, OxidationStateReward

# Optoelectronic / DFT-derived properties
from src.rewards.bandgap_reward import BandGapReward
from src.rewards.effective_mass_reward import EffectiveMassReward
from src.rewards.optical_absorption_reward import OpticalAbsorptionReward
from src.rewards.dielectric_reward import DielectricReward
from src.rewards.defect_reward import DefectReward
from src.rewards.thermal_conductivity_reward import ThermalConductivityReward
from src.rewards.phonon_reward import PhononReward

__all__ = [
    # Core
    "RewardComponent",
    "CompositeReward",
    # Stability
    "EhullReward",
    # Validity / diversity / composition
    "ValidityReward",
    "DiversityReward",
    "NoveltyReward",
    "CompositionReward",
    "OxidationStateReward",
    # Optoelectronic properties
    "BandGapReward",
    "EffectiveMassReward",
    "OpticalAbsorptionReward",
    "DielectricReward",
    # Defect tolerance
    "DefectReward",
    # Thermal / phonon
    "ThermalConductivityReward",
    "PhononReward",
]
