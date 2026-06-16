# CLOVER-Mat

**Reinforcement Learning for Crystal Structure Generation**  
Guided by DFT-derived optoelectronic properties as multi-objective reward signals.  
Built on top of [Chemeleon2](https://github.com/hspark1212/chemeleon2) (VAE + LDM backbone).

---

## Architecture Overview

```
CLOVER-Mat/
├── src/
│   ├── generator/        # Structure generation interface (wraps Chemeleon2 LDM)
│   ├── rewards/          # Reward components (see table below)
│   ├── rl/               # RL training loop (GRPO / PPO policy optimization)
│   ├── evaluators/       # Offline evaluation utilities
│   └── utils/            # Logging, seeding, mp-api helpers
├── custom_rewards/       # Drop-in custom reward plugins (user-defined)
├── configs/
│   └── experiments/      # Named experiment YAML presets
├── scripts/              # Entry-point CLI scripts
├── tests/                # Unit & integration tests
└── notebooks/            # Tutorial notebooks
```

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/lits19/CLOVER-Mat
cd CLOVER-Mat
pip install -e .
```

### 2. Set up Materials Project API key

```bash
export MP_API_KEY="your_key_here"
```

### 3. Run an experiment

**Solar cell absorbers** (band gap 1.0–1.8 eV, high dielectric, defect-tolerant):

```bash
python scripts/train_rl.py \
    --config configs/experiments/optoelectronic_solar.yaml
```

**Thermoelectric materials** (narrow gap, low κ_L):

```bash
python scripts/train_rl.py \
    --config configs/experiments/thermoelectric.yaml
```

**Ag-Bi-I stability baseline** (E_hull only):

```bash
python scripts/train_rl.py \
    --config configs/experiments/ag_bi_i_ehull.yaml
```

### 4. Sample without RL (baseline)

```bash
python scripts/sample.py \
    --elements Pb Sn I Br Cs \
    --n_structures 100 \
    --output outputs/baseline.json
```

---

## Reward System

Rewards are **composable**. Each `RewardComponent` returns a `torch.Tensor` of shape `(N,)`, normalized and weighted before summing.

```python
from src.rewards import CompositeReward, BandGapReward, EhullReward, PhononReward

reward_fn = CompositeReward(components=[
    EhullReward(weight=1.5, backend="mp_api"),
    BandGapReward(weight=2.0, target_min=1.0, target_max=1.8),
    PhononReward(weight=1.0, reward_type="stability"),
])

rewards = reward_fn(structures)  # torch.Tensor of shape (N,)
```

Or configure in YAML:

```yaml
reward:
  components:
    - target: src.rewards.bandgap_reward.BandGapReward
      weight: 2.0
      target_min: 1.0
      target_max: 1.8
    - target: src.rewards.ehull_reward.EhullReward
      weight: 1.5
```

---

## Built-in Reward Components

### Stability

| Component | File | Description |
|---|---|---|
| `EhullReward` | `src/rewards/ehull_reward.py` | Energy above convex hull; backends: MP API, MACE |

### Validity / Diversity / Composition

| Component | File | Description |
|---|---|---|
| `ValidityReward` | `src/rewards/validity_reward.py` | Charge neutrality + electronegativity (SMACT) |
| `DiversityReward` | `src/rewards/diversity_reward.py` | Structural fingerprint diversity |
| `NoveltyReward` | `src/rewards/diversity_reward.py` | Distance to known MP structures |
| `CompositionReward` | `src/rewards/composition_reward.py` | Required element constraints |
| `OxidationStateReward` | `src/rewards/composition_reward.py` | Preferred oxidation state targeting |

### Optoelectronic Properties

| Component | File | Description | Backends |
|---|---|---|---|
| `BandGapReward` | `src/rewards/bandgap_reward.py` | Target a band gap range (eV) | MP API, heuristic |
| `EffectiveMassReward` | `src/rewards/effective_mass_reward.py` | Carrier effective mass m* (units of m₀) | MP API, band structure curvature |
| `OpticalAbsorptionReward` | `src/rewards/optical_absorption_reward.py` | Integrated α(ω)·φ_AM1.5 in visible range; SLME | MP dielectric function |
| `DielectricReward` | `src/rewards/dielectric_reward.py` | Static / electronic / ionic dielectric constant | MP API, Clausius-Mossotti |

### Defect Properties

| Component | File | Description | Backends |
|---|---|---|---|
| `DefectReward` | `src/rewards/defect_reward.py` | Intrinsic defect tolerance score; deep-trap penalty | heuristic, doped JSON files, MP |

To connect pre-computed [doped](https://doped.readthedocs.io) / [ShakeNBreak](https://shakenbreak.readthedocs.io) DFT results:

```yaml
- target: src.rewards.defect_reward.DefectReward
  backend: dft_results
  dft_results_dir: data/defect_results/   # {formula}.json files
```

Use `generate_doped_input()` to batch-generate VASP inputs for the best RL candidates:

```python
from src.rewards.defect_reward import generate_doped_input
generate_doped_input(top_structures, output_dir="vasp_defect_runs/")
```

### Thermal / Phonon Properties

| Component | File | Description | Backends |
|---|---|---|---|
| `ThermalConductivityReward` | `src/rewards/thermal_conductivity_reward.py` | Lattice thermal conductivity κ_L (W/mK); thermoelectric or thermal-management mode | MP Phono3py, Slack-Debye model |
| `PhononReward` | `src/rewards/phonon_reward.py` | Dynamical stability (no imaginary modes), Debye temperature, minimum phonon frequency | MP, MACE+phonopy, heuristic |

---

## Experiment Configs

| Config | Target application | Key rewards |
|---|---|---|
| [`ag_bi_i_ehull.yaml`](configs/experiments/ag_bi_i_ehull.yaml) | Ag-Bi-I stability baseline | E_hull, validity, composition |
| [`optoelectronic_solar.yaml`](configs/experiments/optoelectronic_solar.yaml) | Halide perovskite solar absorbers | Band gap, m*, ε∞, absorption, defect tolerance, phonon stability |
| [`thermoelectric.yaml`](configs/experiments/thermoelectric.yaml) | Low-κ thermoelectric chalcogenides | E_hull, narrow band gap, κ_L, phonon stability |

---

## Adding Custom Rewards

1. Create a file in `custom_rewards/` or `src/rewards/`
2. Subclass `RewardComponent` and implement `compute()`
3. Reference in a YAML config

```python
# custom_rewards/my_reward.py
from src.rewards.base import RewardComponent
import torch

class MyReward(RewardComponent):
    weight: float = 1.0
    normalize: str = "zscore"

    def compute(self, structures, **kwargs) -> torch.Tensor:
        values = [my_property(s) for s in structures]
        return torch.tensor(values, dtype=torch.float32)
```

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
