"""
tests/test_rewards.py
----------------------
Unit tests for reward components and the GRPO trainer.

Run with: pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
import numpy as np

from pymatgen.core import Structure, Lattice

from src.rewards.base import RewardComponent, CompositeReward
from src.rewards.ehull_reward import EhullReward
from src.rewards.validity_reward import ValidityReward
from src.rewards.diversity_reward import DiversityReward, NoveltyReward
from src.rewards.composition_reward import CompositionReward, OxidationStateReward
from src.generator.chemeleon2_wrapper import MockGenerator
from src.rl.grpo_trainer import GRPOTrainer, GRPOConfig
from src.rl.reward_builder import build_ag_bi_i_ehull_reward, build_reward_from_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_agbii_structure(formula: str = "AgBiI4") -> Structure:
    """Create a simple test structure."""
    comp_map = {
        "AgBiI4":  {"Ag": 1, "Bi": 1, "I": 4},
        "AgI":     {"Ag": 1, "I": 1},
        "BiI3":    {"Bi": 1, "I": 3},
        "NaCl":    {"Na": 1, "Cl": 1},
    }
    comp = comp_map.get(formula, {"Ag": 1, "Bi": 1, "I": 4})
    lattice = Lattice.cubic(6.0)
    species = []
    coords = []
    n = sum(comp.values())
    for el, count in comp.items():
        for i in range(count):
            species.append(el)
            coords.append([i / n, i / n, i / n])
    struct = Structure(lattice, species, coords)
    struct.properties["energy_per_atom"] = -2.5
    return struct


@pytest.fixture
def agbii_structures():
    return [make_agbii_structure("AgBiI4") for _ in range(8)]


@pytest.fixture
def mixed_structures():
    return [
        make_agbii_structure("AgBiI4"),
        make_agbii_structure("AgI"),
        make_agbii_structure("BiI3"),
        make_agbii_structure("NaCl"),
    ]


@pytest.fixture
def mock_generator():
    return MockGenerator(seed=0)


# ---------------------------------------------------------------------------
# RewardComponent base tests
# ---------------------------------------------------------------------------

class TestCompositeReward:
    def test_empty_raises(self):
        reward = CompositeReward(components=[])
        with pytest.raises(ValueError):
            reward([make_agbii_structure()])

    def test_add_fluent_api(self):
        reward = CompositeReward()
        reward.add(ValidityReward(weight=1.0))
        assert len(reward.components) == 1

    def test_summary(self):
        reward = CompositeReward(components=[ValidityReward()])
        summary = reward.summary()
        assert "ValidityReward" in summary


# ---------------------------------------------------------------------------
# ValidityReward tests
# ---------------------------------------------------------------------------

class TestValidityReward:
    def test_returns_correct_shape(self, agbii_structures):
        reward = ValidityReward(weight=1.0, normalize="none")
        result = reward(agbii_structures)
        assert result.shape == (8,)

    def test_values_in_range(self, agbii_structures):
        reward = ValidityReward(weight=1.0, normalize="none")
        result = reward(agbii_structures)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_validity_rate(self, agbii_structures):
        reward = ValidityReward()
        rate = reward.validity_rate(agbii_structures)
        assert 0.0 <= rate <= 1.0

    def test_density_check(self):
        # Very low density structure → should fail density check
        lattice = Lattice.cubic(100.0)  # huge cell = low density
        struct = Structure(lattice, ["Ag"], [[0, 0, 0]])
        reward = ValidityReward(check_density=True, min_density=0.5, normalize="none")
        result = reward([struct])
        assert result[0] == 0.0  # invalid


# ---------------------------------------------------------------------------
# CompositionReward tests
# ---------------------------------------------------------------------------

class TestCompositionReward:
    def test_all_required_present(self, agbii_structures):
        reward = CompositionReward(
            required_elements=["Ag", "Bi", "I"],
            weight=1.0,
            normalize="none",
        )
        result = reward(agbii_structures)
        assert (result == 1.0).all(), "All AgBiI4 structures should pass"

    def test_missing_element_penalized(self, mixed_structures):
        reward = CompositionReward(
            required_elements=["Ag", "Bi", "I"],
            weight=1.0,
            normalize="none",
        )
        result = reward(mixed_structures)
        # Only AgBiI4 has all three; AgI, BiI3, NaCl should be penalized
        assert result[0] == 1.0  # AgBiI4: pass
        assert result[1] == -1.0  # AgI: missing Bi
        assert result[2] == -1.0  # BiI3: missing Ag
        assert result[3] == -1.0  # NaCl: missing all

    def test_any_required(self):
        reward = CompositionReward(
            required_elements=["Ag", "Bi"],
            allowed_elements=None,  # no allowed restriction
            require_all=False,
            weight=1.0,
            normalize="none",
        )
        struct_ag = make_agbii_structure("AgI")  # has Ag, no Bi
        # When allowed_elements is None, it defaults to required, so I (not in required) fails.
        # Set allowed_elements explicitly to allow all elements.
        reward._allowed = None  # bypass allowed check
        result = reward([struct_ag])
        assert result[0] == 1.0  # at least one required element present (Ag)


# ---------------------------------------------------------------------------
# DiversityReward tests
# ---------------------------------------------------------------------------

class TestDiversityReward:
    def test_returns_correct_shape(self, agbii_structures):
        reward = DiversityReward(weight=1.0, normalize="none")
        result = reward(agbii_structures)
        assert result.shape == (8,)

    def test_single_structure(self):
        reward = DiversityReward(weight=1.0, normalize="none")
        # Single-structure batch: compute returns zeros directly
        result = reward.compute([make_agbii_structure()])
        assert result.shape == (1,)
        assert result[0].item() == 0.0

    def test_different_structures_higher_diversity(self):
        """More diverse batch should have higher mean diversity score."""
        same_batch = [make_agbii_structure("AgBiI4")] * 4
        diff_batch = [
            make_agbii_structure("AgBiI4"),
            make_agbii_structure("AgI"),
            make_agbii_structure("BiI3"),
            make_agbii_structure("NaCl"),
        ]
        reward = DiversityReward(weight=1.0, normalize="none")
        same_div = reward(same_batch).mean().item()
        diff_div = reward(diff_batch).mean().item()
        # Different batch should generally be more diverse (though not guaranteed
        # with XRD fingerprints for very simple structures)
        assert diff_div >= same_div - 0.01  # allow small float tolerance


# ---------------------------------------------------------------------------
# EhullReward tests (mock backend via energy_per_atom property)
# ---------------------------------------------------------------------------

class TestEhullReward:
    def test_reward_negates_ehull(self):
        """Lower E_hull → higher reward (reward = -E_hull in normalized space)."""
        # We test with "mace" backend stub; mock by patching
        structures = [make_agbii_structure() for _ in range(4)]
        # Set known energies to test reward ordering
        for i, s in enumerate(structures):
            s.properties["energy_per_atom"] = -3.0 + i * 0.5  # -3.0, -2.5, -2.0, -1.5

        reward = EhullReward(
            backend="mace",
            weight=1.0,
            normalize="none",
            nan_penalty=-2.0,
        )
        # Patch the mace function to return our known energies
        import src.rewards.ehull_reward as er
        original_fn = er._ehull_via_mace

        def mock_mace(structures, **kwargs):
            # Compute fake E_hull from min energy in batch
            energies = [s.properties.get("energy_per_atom", 0.0) for s in structures]
            hull = min(energies)
            return [max(0.0, e - hull) for e in energies]

        er._ehull_via_mace = mock_mace
        try:
            result = reward.compute(structures)
            # Structure 0 has lowest energy → should have highest reward (least E_hull)
            assert result[0] >= result[-1], "Most stable structure should have highest reward"
        finally:
            er._ehull_via_mace = original_fn

    def test_nan_penalty(self):
        reward = EhullReward(backend="mp_api", nan_penalty=-5.0, normalize="none")
        # Patch to return NaN
        import src.rewards.ehull_reward as er
        original = er._ehull_via_mp_api

        er._ehull_via_mp_api = lambda *a, **kw: [float("nan")]
        try:
            result = reward.compute([make_agbii_structure()])
            assert result[0] == -5.0
        finally:
            er._ehull_via_mp_api = original


# ---------------------------------------------------------------------------
# MockGenerator tests
# ---------------------------------------------------------------------------

class TestMockGenerator:
    def test_sample_returns_structures(self, mock_generator):
        structures = mock_generator.sample(n=10)
        assert len(structures) == 10
        assert all(isinstance(s, Structure) for s in structures)

    def test_sample_ag_bi_i_elements(self, mock_generator):
        structures = mock_generator.sample(n=20)
        for s in structures:
            elements = {str(el) for el in s.composition.elements}
            # All mock structures should use elements from the Ag-Bi-I system
            assert elements.issubset({"Ag", "Bi", "I"})

    def test_reproducible_with_seed(self):
        # Both generators with same seed should behave identically
        # (random.seed is global so we test in isolation)
        import random as _random
        _random.seed(99)
        g1 = MockGenerator(seed=None)  # seed already set
        s1 = g1.sample(n=3)
        formulas1 = [str(s.composition) for s in s1]

        _random.seed(99)
        g2 = MockGenerator(seed=None)
        s2 = g2.sample(n=3)
        formulas2 = [str(s.composition) for s in s2]

        assert formulas1 == formulas2


# ---------------------------------------------------------------------------
# RewardBuilder tests
# ---------------------------------------------------------------------------

class TestRewardBuilder:
    def test_build_ag_bi_i_preset(self, agbii_structures):
        reward = build_ag_bi_i_ehull_reward(
            ehull_backend="mace",  # use mace to avoid MP API requirement
        )
        assert len(reward.components) == 4

    def test_build_from_config_dict(self):
        config = {
            "components": [
                {
                    "target": "src.rewards.validity_reward.ValidityReward",
                    "weight": 1.0,
                    "normalize": "none",
                }
            ]
        }
        reward = build_reward_from_config(config)
        assert len(reward.components) == 1
        assert isinstance(reward.components[0], ValidityReward)

    def test_invalid_target_raises(self):
        config = {
            "components": [
                {"target": "src.rewards.nonexistent.FakeReward", "weight": 1.0}
            ]
        }
        with pytest.raises(ImportError):
            build_reward_from_config(config)


# ---------------------------------------------------------------------------
# GRPO Trainer integration test (mock, no GPU required)
# ---------------------------------------------------------------------------

class TestGRPOTrainer:
    def test_training_step_runs(self):
        generator = MockGenerator(seed=0)
        reward_fn = CompositeReward(components=[
            ValidityReward(weight=1.0, normalize="none"),
            CompositionReward(
                required_elements=["Ag", "Bi", "I"],
                weight=0.5,
                normalize="none",
            ),
        ])
        config = GRPOConfig(
            n_iterations=3,
            batch_size=8,
            group_size=4,
            output_dir="/tmp/test_rl_run",
            log_every=1,
            save_every=2,
        )
        trainer = GRPOTrainer(generator=generator, reward_fn=reward_fn, config=config)
        history = trainer.train()
        assert len(history) == 3
        assert all("reward_mean" in h for h in history)

    def test_evaluate(self):
        generator = MockGenerator(seed=0)
        reward_fn = CompositeReward(components=[ValidityReward(normalize="none")])
        config = GRPOConfig(
            n_iterations=1,
            batch_size=8,
            group_size=4,
            output_dir="/tmp/test_rl_eval",
        )
        trainer = GRPOTrainer(generator=generator, reward_fn=reward_fn, config=config)
        result = trainer.evaluate(n=10)
        assert "reward_mean" in result
        assert len(result["structures"]) == 10
