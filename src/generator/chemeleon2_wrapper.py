"""
src/generator/chemeleon2_wrapper.py
------------------------------------
Wrapper around Chemeleon2's Latent Diffusion Model (LDM) for generating
crystal structures. Also provides a MockGenerator for unit testing without
a GPU or Chemeleon2 installation.

The RL trainer calls:
    structures = generator.sample(n=batch_size, condition=condition_dict)

Which returns a list of pymatgen Structure objects.
"""

from __future__ import annotations

import abc
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch
from pymatgen.core import Structure, Lattice


# ---------------------------------------------------------------------------
# Abstract base generator
# ---------------------------------------------------------------------------

class BaseGenerator(abc.ABC):
    """
    Abstract base for structure generators.
    Subclass this to plug in any backbone model.
    """

    @abc.abstractmethod
    def sample(
        self,
        n: int,
        condition: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Structure]:
        """Generate n crystal structures."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Return parameters updated by the RL optimizer."""
        raise NotImplementedError

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        pass

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        pass


# ---------------------------------------------------------------------------
# Chemeleon2 LDM wrapper
# ---------------------------------------------------------------------------

class Chemeleon2Generator(BaseGenerator):
    """
    Wraps Chemeleon2's LDM module.

    Parameters
    ----------
    checkpoint_path : str
        Path to a pretrained LDM checkpoint (.ckpt).
    vae_checkpoint_path : str
        Path to a pretrained VAE checkpoint (.ckpt).
    device : str
        "cuda" | "cpu" | "mps"
    n_diffusion_steps : int
        Number of reverse diffusion steps for sampling.
    guidance_scale : float
        Classifier-free guidance scale.
    """

    def __init__(
        self,
        checkpoint_path: str,
        vae_checkpoint_path: str,
        device: str = "cuda",
        n_diffusion_steps: int = 1000,
        guidance_scale: float = 1.0,
    ):
        self.device = device
        self.n_diffusion_steps = n_diffusion_steps
        self.guidance_scale = guidance_scale

        self._ldm = self._load_ldm(checkpoint_path, vae_checkpoint_path)

    def _load_ldm(self, ldm_path: str, vae_path: str):
        """Load Chemeleon2 LDM from checkpoints."""
        try:
            # Chemeleon2 uses Lightning modules; load accordingly
            from src.ldm_module.ldm_module import LDMModule  # chemeleon2 src path
            from src.vae_module.vae_module import VAEModule

            vae = VAEModule.load_from_checkpoint(vae_path, map_location=self.device)
            vae.eval()

            ldm = LDMModule.load_from_checkpoint(
                ldm_path, vae=vae, map_location=self.device
            )
            ldm.eval()
            ldm.to(self.device)
            return ldm

        except ImportError:
            raise ImportError(
                "Chemeleon2 not found. Install with:\n"
                "  pip install 'rl-crystal-gen[chemeleon2]'\n"
                "or: pip install git+https://github.com/hspark1212/chemeleon2"
            )

    def sample(
        self,
        n: int,
        condition: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Structure]:
        with torch.no_grad():
            # Chemeleon2 LDM sample interface
            structures = self._ldm.sample(
                num_samples=n,
                num_steps=self.n_diffusion_steps,
                guidance_scale=self.guidance_scale,
                condition=condition,
            )
        return structures

    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        return list(self._ldm.parameters())

    def save_checkpoint(self, path: str) -> None:
        torch.save(self._ldm.state_dict(), path)

    def load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self._ldm.load_state_dict(state)


# ---------------------------------------------------------------------------
# Mock generator for testing without GPU / Chemeleon2
# ---------------------------------------------------------------------------

# Common oxidation states for Ag-Bi-I
_MOCK_FORMULAS = [
    "AgBiI4", "Ag3BiI6", "AgBi2I7", "Ag2BiI5",
    "Bi3I", "BiI3", "AgI", "Ag3Bi2I9",
]

_MOCK_SPACEGROUPS = [1, 12, 62, 63, 139, 166, 194, 225]  # common SGs


def _make_mock_structure(formula: str = "AgBiI4") -> Structure:
    """Generate a random plausible crystal structure for testing."""
    from pymatgen.core import Element

    comp_map = {
        "AgBiI4":   {"Ag": 1, "Bi": 1, "I": 4},
        "Ag3BiI6":  {"Ag": 3, "Bi": 1, "I": 6},
        "AgBi2I7":  {"Ag": 1, "Bi": 2, "I": 7},
        "Ag2BiI5":  {"Ag": 2, "Bi": 1, "I": 5},
        "BiI3":     {"Bi": 1, "I": 3},
        "AgI":      {"Ag": 1, "I": 1},
        "Ag3Bi2I9": {"Ag": 3, "Bi": 2, "I": 9},
    }

    comp = comp_map.get(formula, {"Ag": 1, "Bi": 1, "I": 4})

    # Random cubic-ish lattice (a between 5 and 15 Å)
    a = random.uniform(5.0, 15.0)
    b = random.uniform(5.0, 15.0)
    c = random.uniform(5.0, 15.0)
    alpha = random.uniform(70.0, 110.0)
    beta  = random.uniform(70.0, 110.0)
    gamma = random.uniform(70.0, 110.0)
    lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)

    species = []
    coords = []
    for el, count in comp.items():
        for _ in range(count):
            species.append(el)
            coords.append(np.random.rand(3))

    struct = Structure(lattice, species, coords)
    # Attach a fake DFT energy for testing E_hull reward
    struct.properties["energy_per_atom"] = random.uniform(-3.5, -1.0)
    return struct


@dataclass
class MockGenerator(BaseGenerator):
    """
    Generates random Ag-Bi-I structures for unit testing.
    Does not require any ML model or GPU.

    Parameters
    ----------
    formulas : list[str]
        Pool of chemical formulas to sample from.
    noise_scale : float
        Positional noise added to atomic coordinates.
    seed : int | None
        Random seed for reproducibility.
    """

    formulas: List[str] = field(default_factory=lambda: _MOCK_FORMULAS)
    noise_scale: float = 0.05
    seed: Optional[int] = None

    def __post_init__(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

    # Fake parameters for RL optimizer testing
    _fake_params: List[torch.nn.Parameter] = field(
        default_factory=lambda: [torch.nn.Parameter(torch.zeros(1))],
        repr=False,
    )

    def sample(
        self,
        n: int,
        condition: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Structure]:
        structures = []
        for _ in range(n):
            formula = random.choice(self.formulas)
            struct = _make_mock_structure(formula)
            structures.append(struct)
        return structures

    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        return self._fake_params
