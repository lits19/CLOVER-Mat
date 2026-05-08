# CLOVER-Mat

**Reinforcement Learning for Crystal Structure Generation**  
Guided by E_hull (energy above convex hull) and other self-defined properties as the reward signal.  
Built on top of [Chemeleon2](https://github.com/hspark1212/chemeleon2) (VAE + LDM backbone).

---

## Architecture Overview

```
rl-crystal-gen/
├── src/
│   ├── generator/        # Structure generation interface (wraps Chemeleon2 LDM)
│   ├── rewards/          # Reward components (E_hull, validity, diversity, …)
│   ├── rl/               # RL training loop (GRPO / PPO policy optimization)
│   ├── evaluators/       # Offline evaluation utilities
│   └── utils/            # Logging, seeding, mp-api helpers
├── custom_rewards/       # Drop-in custom reward plugins (user-defined)
├── configs/              # Hydra YAML configs
│   ├── experiments/      # Named experiment presets
│   └── rewards/          # Reward composition configs
├── scripts/              # Entry-point CLI scripts
├── tests/                # Unit & integration tests
├── notebooks/            # Tutorial notebooks
└── docs/                 # Extended documentation
```

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/lits19/rl-crystal-gen
cd rl-crystal-gen
pip install -e .
```

### 2. Set up Materials Project API key

```bash
export MP_API_KEY="your_key_here"
```

### 3. Run structure generation for Ag-Bi-I with E_hull reward

```bash
python scripts/train_rl.py \
    --config configs/experiments/ag_bi_i_ehull.yaml
```

### 4. Sample without RL (baseline)

```bash
python scripts/sample.py \
    --elements Ag Bi I \
    --n_structures 100 \
    --output outputs/ag_bi_i_baseline.json
```

---

## Reward System

Rewards are **composable**. Each `RewardComponent` returns a `torch.Tensor` of shape `(N,)`.

```python
# custom_rewards/my_reward.py
from src.rewards.base import RewardComponent
import torch

class MyReward(RewardComponent):
    weight: float = 1.0

    def compute(self, structures, **kwargs) -> torch.Tensor:
        values = [some_property(s) for s in structures]
        return torch.tensor(values, dtype=torch.float32)
```

Configure in YAML:

```yaml
reward:
  components:
    - target: src.rewards.ehull_reward.EhullReward
      weight: 1.0
    - target: custom_rewards.my_reward.MyReward
      weight: 0.5
```

---

## Built-in Reward Components

| Component | File | Description |
|---|---|---|
| `EhullReward` | `src/rewards/ehull_reward.py` | Energy above convex hull (MP API or MACE) |
| `ValidityReward` | `src/rewards/validity_reward.py` | Charge neutrality + electronegativity |
| `DiversityReward` | `src/rewards/diversity_reward.py` | Structural fingerprint diversity |
| `NoveltyReward` | `src/rewards/novelty_reward.py` | Distance to known structures in MP |
| `CompositionReward` | `src/rewards/composition_reward.py` | Target element constraint |

---

## Adding New Modules

1. **New reward** → add file in `custom_rewards/` or `src/rewards/`
2. **New evaluator** → add file in `src/evaluators/`
3. **New RL algorithm** → subclass `src/rl/base_trainer.py`
4. **New experiment** → add YAML in `configs/experiments/`

---

## Citation

If you use this work, please cite Chemeleon2:

```bibtex
@article{Park2025chemeleon2,
  title={Guiding Generative Models to Uncover Diverse and Novel Crystals via Reinforcement Learning},
  author={Hyunsoo Park and Aron Walsh},
  year={2025},
  url={https://arxiv.org/abs/2511.07158}
}
```
