"""
src/rewards/base.py
-------------------
Abstract base class for all reward components.

Every reward component must:
  - Subclass RewardComponent
  - Implement compute(structures, **kwargs) -> torch.Tensor of shape (N,)
  - Declare a `weight` float attribute (default 1.0)

Rewards are normalized per batch before weighting (configurable).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from pymatgen.core import Structure


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def normalize_zscore(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Zero-mean, unit-variance normalization."""
    return (x - x.mean()) / (x.std() + eps)


def normalize_minmax(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min() + eps)


def normalize_none(x: torch.Tensor, **_) -> torch.Tensor:
    return x


NORMALIZERS = {
    "zscore": normalize_zscore,
    "minmax": normalize_minmax,
    "none": normalize_none,
}


# ---------------------------------------------------------------------------
# Base component
# ---------------------------------------------------------------------------

@dataclass
class RewardComponent(abc.ABC):
    """
    Base class for a scalar reward signal over a batch of crystal structures.

    Attributes
    ----------
    weight : float
        Linear weight applied after normalization.
    normalize : str
        One of "zscore", "minmax", "none".
    clip_min, clip_max : float | None
        Optional clipping of raw values before normalization.
    """

    weight: float = 1.0
    normalize: str = "zscore"
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None

    @abc.abstractmethod
    def compute(self, structures: List[Structure], **kwargs) -> torch.Tensor:
        """
        Compute raw reward values.

        Parameters
        ----------
        structures : list[pymatgen.core.Structure]
            A batch of generated crystal structures.
        **kwargs : dict
            Extra context passed by the RL trainer (e.g., step, composition).

        Returns
        -------
        torch.Tensor of shape (N,), dtype float32
        """
        raise NotImplementedError

    def __call__(self, structures: List[Structure], **kwargs) -> torch.Tensor:
        raw = self.compute(structures, **kwargs).float()

        # Optional clipping
        if self.clip_min is not None or self.clip_max is not None:
            raw = torch.clamp(raw, min=self.clip_min, max=self.clip_max)

        # Normalize
        norm_fn = NORMALIZERS.get(self.normalize, normalize_none)
        normed = norm_fn(raw)

        return self.weight * normed


# ---------------------------------------------------------------------------
# Composite reward (weighted sum of components)
# ---------------------------------------------------------------------------

@dataclass
class CompositeReward:
    """
    Aggregates multiple RewardComponent instances into a single scalar reward.

    Usage
    -----
    reward_fn = CompositeReward(components=[
        EhullReward(weight=1.0),
        ValidityReward(weight=0.5),
    ])
    rewards = reward_fn(structures)  # shape (N,)
    """

    components: List[RewardComponent] = field(default_factory=list)

    def __call__(self, structures: List[Structure], **kwargs) -> torch.Tensor:
        if not self.components:
            raise ValueError("CompositeReward has no components.")

        total = torch.zeros(len(structures), dtype=torch.float32)
        for component in self.components:
            total = total + component(structures, **kwargs)

        return total

    def add(self, component: RewardComponent) -> "CompositeReward":
        """Fluent API: reward_fn.add(MyReward(weight=0.5))"""
        self.components.append(component)
        return self

    def summary(self) -> str:
        lines = ["CompositeReward:"]
        for c in self.components:
            lines.append(f"  {c.__class__.__name__}(weight={c.weight}, normalize={c.normalize})")
        return "\n".join(lines)
