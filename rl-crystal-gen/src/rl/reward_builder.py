"""
src/rl/reward_builder.py
-------------------------
Build a CompositeReward from a configuration dictionary or YAML file.
Supports dynamic class loading so users can register custom reward components
without modifying core source files.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List

from src.rewards.base import CompositeReward, RewardComponent

logger = logging.getLogger(__name__)


def _load_class(target: str) -> type:
    """
    Load a class from a dotted path string.
    Example: "src.rewards.ehull_reward.EhullReward"
    """
    parts = target.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid target path: '{target}'. Use 'module.ClassName'")
    module_path, class_name = parts
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Cannot load '{target}': {e}") from e


def build_reward_from_config(config: Dict[str, Any]) -> CompositeReward:
    """
    Build a CompositeReward from a configuration dict.

    Expected format
    ---------------
    {
        "components": [
            {
                "target": "src.rewards.ehull_reward.EhullReward",
                "weight": 1.0,
                "normalize": "zscore",
                "backend": "mace",
                ...  # any kwargs for the reward class
            },
            {
                "target": "src.rewards.validity_reward.ValidityReward",
                "weight": 0.5
            }
        ]
    }

    Parameters
    ----------
    config : dict
        Reward configuration dict.

    Returns
    -------
    CompositeReward
    """
    components_config = config.get("components", [])
    if not components_config:
        raise ValueError("Reward config must have at least one component under 'components'.")

    components: List[RewardComponent] = []
    for comp_cfg in components_config:
        comp_cfg = dict(comp_cfg)  # copy
        target = comp_cfg.pop("target")
        cls = _load_class(target)

        # Validate it's a RewardComponent
        if not issubclass(cls, RewardComponent):
            raise TypeError(f"{target} must subclass RewardComponent")

        instance = cls(**comp_cfg)
        components.append(instance)
        logger.info(f"  Loaded reward: {cls.__name__}(weight={instance.weight})")

    return CompositeReward(components=components)


def build_reward_from_yaml(yaml_path: str) -> CompositeReward:
    """
    Build a CompositeReward from a YAML file.

    Parameters
    ----------
    yaml_path : str
        Path to a YAML configuration file.

    Returns
    -------
    CompositeReward
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("Install pyyaml: pip install pyyaml")

    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Reward config not found: {yaml_path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    reward_config = config.get("reward", config)  # support both flat and nested
    return build_reward_from_config(reward_config)


# ---------------------------------------------------------------------------
# Pre-built reward configs for common use cases
# ---------------------------------------------------------------------------

def build_ag_bi_i_ehull_reward(
    ehull_backend: str = "mp_api",
    ehull_weight: float = 1.0,
    validity_weight: float = 0.5,
    composition_weight: float = 0.5,
    diversity_weight: float = 0.3,
) -> CompositeReward:
    """
    Pre-built composite reward for Ag-Bi-I ternary structure generation.

    Reward components:
      1. EhullReward       – optimize thermodynamic stability
      2. ValidityReward    – enforce charge neutrality & geometry
      3. CompositionReward – ensure Ag, Bi, I are present
      4. DiversityReward   – encourage structural diversity

    Parameters
    ----------
    ehull_backend : str
        "mp_api" or "mace"
    ehull_weight : float
    validity_weight : float
    composition_weight : float
    diversity_weight : float

    Returns
    -------
    CompositeReward
    """
    from src.rewards.ehull_reward import EhullReward
    from src.rewards.validity_reward import ValidityReward
    from src.rewards.composition_reward import CompositionReward
    from src.rewards.diversity_reward import DiversityReward

    reward = CompositeReward(components=[
        EhullReward(
            backend=ehull_backend,
            weight=ehull_weight,
            normalize="zscore",
            nan_penalty=-2.0,
            stable_threshold=0.1,
        ),
        ValidityReward(
            weight=validity_weight,
            normalize="none",
        ),
        CompositionReward(
            required_elements=["Ag", "Bi", "I"],
            weight=composition_weight,
            normalize="none",
        ),
        DiversityReward(
            weight=diversity_weight,
            normalize="zscore",
        ),
    ])

    logger.info("Built Ag-Bi-I composite reward:")
    logger.info(reward.summary())
    return reward
