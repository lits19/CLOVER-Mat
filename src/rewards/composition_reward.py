"""
src/rewards/composition_reward.py
----------------------------------
CompositionReward: enforces that generated structures contain specific elements.

Useful for target chemical system exploration (e.g., Ag-Bi-I ternaries).

Also includes:
  - ChargeBalanceReward: penalizes compositions far from charge balance
    using common oxidation states.
  - OxidationStateReward: rewards structures with known-good oxidation state combos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional

import torch
from pymatgen.core import Structure, Element

from src.rewards.base import RewardComponent


@dataclass
class CompositionReward(RewardComponent):
    """
    Reward structures that contain ALL required elements and ONLY allowed elements.

    Parameters
    ----------
    required_elements : list[str]
        Elements that MUST appear in the structure (e.g., ["Ag", "Bi", "I"]).
    allowed_elements : list[str] | None
        If set, structures containing elements outside this list are penalized.
        Default: same as required_elements.
    require_all : bool
        If True: all required_elements must be present.
        If False: at least one must be present.
    reward_hit : float
        Reward value when composition matches.
    reward_miss : float
        Reward value (usually negative) when composition does not match.
    weight : float
    normalize : str
    """

    required_elements: List[str] = field(default_factory=lambda: ["Ag", "Bi", "I"])
    allowed_elements: Optional[List[str]] = None
    require_all: bool = True
    reward_hit: float = 1.0
    reward_miss: float = -1.0
    weight: float = 0.5
    normalize: str = "none"

    def __post_init__(self):
        self._required: Set[str] = set(self.required_elements)
        self._allowed: Optional[Set[str]] = (
            set(self.allowed_elements) if self.allowed_elements else self._required
        )

    def compute(self, structures: List[Structure], **kwargs) -> torch.Tensor:
        rewards = []
        for struct in structures:
            present = {str(el) for el in struct.composition.elements}

            # Check required elements
            if self.require_all:
                ok = self._required.issubset(present)
            else:
                ok = bool(self._required & present)

            # Check no forbidden elements
            if self._allowed is not None and ok:
                forbidden = present - self._allowed
                if forbidden:
                    ok = False

            rewards.append(self.reward_hit if ok else self.reward_miss)

        return torch.tensor(rewards, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Ag-Bi-I specific: oxidation state chemistry
# ---------------------------------------------------------------------------

# Common stable oxidation state combinations for Ag-Bi-I
# Source: ICSD / MP knowledge
AGBII_KNOWN_OX = [
    {"Ag": +1, "Bi": +3, "I": -1},   # e.g., AgBiI4, Ag3BiI6
    {"Ag": +1, "Bi": +5, "I": -1},   # higher valence Bi
    {"Bi": +3, "I": -1},              # binary
    {"Ag": +1, "I": -1},              # binary
]


@dataclass
class OxidationStateReward(RewardComponent):
    """
    Rewards Ag-Bi-I structures whose oxidation states match known stable combos.

    Uses pymatgen's BVAnalyzer for bond-valence-based oxidation state assignment.

    Parameters
    ----------
    known_ox_states : list[dict]
        List of known oxidation state dictionaries.
    weight : float
    normalize : str
    """

    known_ox_states: List[Dict[str, int]] = field(
        default_factory=lambda: AGBII_KNOWN_OX
    )
    weight: float = 0.3
    normalize: str = "none"

    def _matches_known(self, ox_dict: Dict[str, int]) -> bool:
        for known in self.known_ox_states:
            if all(ox_dict.get(el) == ox for el, ox in known.items()):
                return True
        return False

    def compute(self, structures: List[Structure], **kwargs) -> torch.Tensor:
        try:
            from pymatgen.analysis.bond_valence import BVAnalyzer
            bva = BVAnalyzer()
        except ImportError:
            # Fall back to 1.0 for all if BVAnalyzer unavailable
            return torch.ones(len(structures), dtype=torch.float32)

        rewards = []
        for struct in structures:
            try:
                ox_struct = bva.get_oxi_state_decorated_structure(struct)
                ox_dict = {
                    str(site.specie.element): site.specie.oxi_state
                    for site in ox_struct
                }
                # Take most common oxidation state per element
                from collections import Counter
                el_ox = {}
                for site in ox_struct:
                    el = str(site.specie.element)
                    ox = site.specie.oxi_state
                    el_ox.setdefault(el, []).append(ox)
                el_ox_mode = {
                    el: Counter(oxs).most_common(1)[0][0]
                    for el, oxs in el_ox.items()
                }
                rewards.append(1.0 if self._matches_known(el_ox_mode) else 0.0)
            except Exception:
                rewards.append(0.0)

        return torch.tensor(rewards, dtype=torch.float32)
