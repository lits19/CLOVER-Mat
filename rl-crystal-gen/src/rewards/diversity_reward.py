"""
src/rewards/diversity_reward.py
-------------------------------
DiversityReward: encourages structural and compositional diversity
within a generated batch using fingerprint-based pairwise distances.

NoveltyReward: penalizes structures already present in the
Materials Project database (or a reference set).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from pymatgen.core import Structure

from src.rewards.base import RewardComponent


# ---------------------------------------------------------------------------
# Structural fingerprint helper
# ---------------------------------------------------------------------------

def _get_fingerprint(structure: Structure, radius: float = 8.0, nbins: int = 64) -> np.ndarray:
    """
    Compute a simple radial distribution function (RDF) fingerprint.
    Falls back to composition-only fingerprint if RDF computation fails.
    """
    try:
        from pymatgen.analysis.diffraction.xrd import XRDCalculator
        xrd = XRDCalculator()
        pattern = xrd.get_pattern(structure)
        # Simple histogram of XRD 2-theta intensities
        hist, _ = np.histogram(
            pattern.x, bins=nbins, range=(0, 90), weights=pattern.y
        )
        norm = np.linalg.norm(hist)
        return hist / (norm + 1e-8)
    except Exception:
        # Fallback: one-hot composition fingerprint
        from pymatgen.core.periodic_table import Element
        fp = np.zeros(118)
        for el, amt in structure.composition.items():
            fp[el.Z - 1] = float(amt)
        norm = np.linalg.norm(fp)
        return fp / (norm + 1e-8)


@dataclass
class DiversityReward(RewardComponent):
    """
    Reward that encourages structural diversity within the generated batch.

    Each structure's reward = its mean pairwise distance to all other
    structures in the batch (higher distance = more novel in batch context).

    Parameters
    ----------
    fingerprint_radius : float
        Cutoff radius for fingerprinting (Å).
    weight : float
    normalize : str
    """

    fingerprint_radius: float = 8.0
    nbins: int = 64
    weight: float = 0.3
    normalize: str = "zscore"

    def compute(self, structures: List[Structure], **kwargs) -> torch.Tensor:
        n = len(structures)
        if n < 2:
            return torch.zeros(n, dtype=torch.float32)

        fps = np.stack([
            _get_fingerprint(s, self.fingerprint_radius, self.nbins)
            for s in structures
        ])  # (N, D)

        # Pairwise cosine similarity -> distance
        norms = np.linalg.norm(fps, axis=1, keepdims=True)
        fps_norm = fps / (norms + 1e-8)
        sim_matrix = fps_norm @ fps_norm.T  # (N, N)
        dist_matrix = 1.0 - sim_matrix     # cosine distance, range [0, 2]

        # Mean distance to all other structures (excluding self)
        np.fill_diagonal(dist_matrix, 0.0)
        mean_dist = dist_matrix.sum(axis=1) / (n - 1)

        return torch.tensor(mean_dist, dtype=torch.float32)


@dataclass
class NoveltyReward(RewardComponent):
    """
    Reward structures that are dissimilar to known Materials Project entries.

    Compares generated structures to a reference set via fingerprint distance.
    Structures with high distance to all reference structures receive high reward.

    Parameters
    ----------
    reference_structures : list[Structure] | None
        Preloaded reference structures. If None, queries MP API for
        relevant entries on first call.
    chemsys : str | None
        Chemical system to query (e.g., "Ag-Bi-I"). Auto-detected if None.
    mp_api_key : str | None
        MP API key (env MP_API_KEY as fallback).
    max_references : int
        Maximum number of reference structures to load.
    novelty_threshold : float
        Cosine distance below which a structure is deemed "known".
    """

    reference_structures: Optional[List[Structure]] = field(default=None, repr=False)
    chemsys: Optional[str] = None
    mp_api_key: Optional[str] = None
    max_references: int = 200
    novelty_threshold: float = 0.05
    weight: float = 0.3
    normalize: str = "minmax"

    def _load_references(self, structures: List[Structure]) -> None:
        """Lazy-load reference structures from MP."""
        if self.reference_structures is not None:
            return

        elements = set()
        for s in structures:
            elements.update([str(el) for el in s.composition.elements])
        chemsys = self.chemsys or "-".join(sorted(elements))

        try:
            from mp_api.client import MPRester
            import os
            key = self.mp_api_key or os.environ.get("MP_API_KEY")
            with MPRester(api_key=key) as mpr:
                docs = mpr.materials.summary.search(
                    chemsys=chemsys,
                    fields=["structure"],
                    num_chunks=1,
                    chunk_size=self.max_references,
                )
                self.reference_structures = [d.structure for d in docs if d.structure]
        except Exception as e:
            warnings.warn(f"Could not load reference structures from MP: {e}")
            self.reference_structures = []

    def compute(self, structures: List[Structure], **kwargs) -> torch.Tensor:
        self._load_references(structures)

        if not self.reference_structures:
            # No references: treat all structures as fully novel
            return torch.ones(len(structures), dtype=torch.float32)

        ref_fps = np.stack([_get_fingerprint(s) for s in self.reference_structures])
        ref_norms = np.linalg.norm(ref_fps, axis=1, keepdims=True)
        ref_fps_norm = ref_fps / (ref_norms + 1e-8)

        novelty_scores = []
        for struct in structures:
            gen_fp = _get_fingerprint(struct)
            gen_norm = np.linalg.norm(gen_fp) + 1e-8
            gen_fp_norm = gen_fp / gen_norm
            # Cosine distances to all reference structures
            sims = ref_fps_norm @ gen_fp_norm  # (n_ref,)
            dists = 1.0 - sims
            min_dist = float(dists.min())
            # Novelty score: 0 if very similar to reference, 1 if very different
            novelty = min(1.0, min_dist / (self.novelty_threshold + 1e-8))
            novelty_scores.append(novelty)

        return torch.tensor(novelty_scores, dtype=torch.float32)
