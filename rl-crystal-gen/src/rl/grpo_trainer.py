"""
src/rl/grpo_trainer.py
-----------------------
GRPO (Group Relative Policy Optimization) trainer for crystal structure generation.

GRPO is the algorithm used by Chemeleon2. It:
  1. Samples G groups of structures from the current policy (LDM).
  2. Computes rewards for each structure.
  3. Computes group-relative advantages: A_i = (r_i - mean(r_group)) / std(r_group).
  4. Updates the policy via a PPO-style clipped objective on the diffusion loss.

Reference: https://arxiv.org/abs/2511.07158 (Chemeleon2)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.rewards.base import CompositeReward
from src.generator.chemeleon2_wrapper import BaseGenerator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GRPO Config
# ---------------------------------------------------------------------------

@dataclass
class GRPOConfig:
    """
    Hyperparameters for GRPO training.

    Parameters
    ----------
    n_iterations : int
        Total number of RL training iterations.
    batch_size : int
        Number of structures sampled per iteration.
    group_size : int
        Number of structures per GRPO group (G in the paper).
        batch_size must be divisible by group_size.
    lr : float
        Learning rate for AdamW.
    clip_epsilon : float
        PPO clipping parameter (ε).
    kl_coeff : float
        KL divergence penalty coefficient (β). Regularizes against
        the reference (pretrained) model.
    entropy_coeff : float
        Entropy bonus coefficient.
    gradient_clip : float
        Max gradient norm for clipping.
    n_update_steps : int
        Number of gradient steps per sampled batch.
    save_every : int
        Save checkpoint every N iterations.
    log_every : int
        Log metrics every N iterations.
    output_dir : str
        Directory for checkpoints and logs.
    device : str
        "cuda" | "cpu" | "mps"
    seed : int
    """

    n_iterations: int = 500
    batch_size: int = 64
    group_size: int = 8
    lr: float = 1e-5
    clip_epsilon: float = 0.2
    kl_coeff: float = 0.01
    entropy_coeff: float = 0.001
    gradient_clip: float = 1.0
    n_update_steps: int = 4
    save_every: int = 50
    log_every: int = 10
    output_dir: str = "outputs/rl_run"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


# ---------------------------------------------------------------------------
# GRPO Trainer
# ---------------------------------------------------------------------------

class GRPOTrainer:
    """
    GRPO Trainer for RL-guided crystal structure generation.

    Usage
    -----
    trainer = GRPOTrainer(
        generator=Chemeleon2Generator(...),
        reward_fn=CompositeReward([EhullReward(), ValidityReward()]),
        config=GRPOConfig(n_iterations=200, batch_size=32),
    )
    trainer.train()
    """

    def __init__(
        self,
        generator: BaseGenerator,
        reward_fn: CompositeReward,
        config: GRPOConfig,
        condition: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[Callable]] = None,
    ):
        """
        Parameters
        ----------
        generator : BaseGenerator
            The structure generator (Chemeleon2 LDM or MockGenerator).
        reward_fn : CompositeReward
            Composite reward function.
        config : GRPOConfig
            Training hyperparameters.
        condition : dict | None
            Optional conditioning dict passed to generator.sample().
            Example: {"elements": ["Ag", "Bi", "I"]}
        callbacks : list[callable] | None
            Optional list of callback functions called after each iteration
            with signature: callback(iteration, metrics, structures, rewards)
        """
        self.generator = generator
        self.reward_fn = reward_fn
        self.config = config
        self.condition = condition or {}
        self.callbacks = callbacks or []

        # Validate group size
        assert config.batch_size % config.group_size == 0, (
            f"batch_size ({config.batch_size}) must be divisible by "
            f"group_size ({config.group_size})"
        )
        self.n_groups = config.batch_size // config.group_size

        # Optimizer
        params = generator.get_trainable_parameters()
        self.optimizer = AdamW(params, lr=config.lr, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=config.n_iterations
        )

        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Metrics history
        self.history: List[Dict[str, float]] = []

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.output_dir / "training.log"),
            ],
        )

        torch.manual_seed(config.seed)
        logger.info(f"GRPOTrainer initialized on device={config.device}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(self.reward_fn.summary())

    # ------------------------------------------------------------------
    # Core training loop
    # ------------------------------------------------------------------

    def train(self) -> List[Dict[str, float]]:
        """Run the full RL training loop."""
        logger.info(
            f"Starting GRPO training for {self.config.n_iterations} iterations"
        )

        for iteration in range(1, self.config.n_iterations + 1):
            metrics = self._training_step(iteration)
            self.history.append(metrics)

            if iteration % self.config.log_every == 0:
                self._log_metrics(iteration, metrics)

            if iteration % self.config.save_every == 0:
                self._save_checkpoint(iteration)

            # Run user callbacks
            for cb in self.callbacks:
                try:
                    cb(iteration, metrics)
                except Exception as e:
                    logger.warning(f"Callback error at iteration {iteration}: {e}")

        logger.info("Training complete.")
        self._save_checkpoint("final")
        return self.history

    def _training_step(self, iteration: int) -> Dict[str, float]:
        """Single GRPO iteration: sample → reward → update."""

        # 1. Sample batch of structures from generator
        structures = self.generator.sample(
            n=self.config.batch_size,
            condition=self.condition,
        )

        # 2. Compute rewards
        with torch.no_grad():
            rewards = self.reward_fn(structures)  # shape (N,)

        # 3. Compute GRPO group-relative advantages
        advantages = self._compute_advantages(rewards)  # shape (N,)

        # 4. Policy gradient update
        policy_loss = self._policy_update(structures, rewards, advantages)

        # 5. Collect metrics
        metrics = {
            "iteration": iteration,
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "reward_max": rewards.max().item(),
            "advantage_mean": advantages.mean().item(),
            "policy_loss": policy_loss,
            "lr": self.scheduler.get_last_lr()[0],
        }

        # Include E_hull stats if available
        ehull_stats = self._extract_ehull_stats(structures)
        metrics.update(ehull_stats)

        self.scheduler.step()
        return metrics

    def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        GRPO: compute group-relative advantages.

        Groups are contiguous blocks of size group_size in the batch.
        A_i = (r_i - mean_group) / (std_group + eps)
        """
        N = len(rewards)
        G = self.config.group_size
        advantages = torch.zeros(N, dtype=torch.float32)

        for g in range(self.n_groups):
            start = g * G
            end = start + G
            group_r = rewards[start:end]
            mean_r = group_r.mean()
            std_r = group_r.std()
            advantages[start:end] = (group_r - mean_r) / (std_r + 1e-8)

        return advantages

    def _policy_update(
        self,
        structures,
        rewards: torch.Tensor,
        advantages: torch.Tensor,
    ) -> float:
        """
        PPO-style clipped policy gradient update.

        For diffusion models, this is applied to the diffusion loss
        weighted by the advantage signal.

        NOTE: For a full Chemeleon2 integration, you would compute the
        LDM denoising loss here and weight it by advantages. This
        implementation provides the framework skeleton that you fill in
        with the actual LDM forward pass.
        """
        total_loss = 0.0
        params = self.generator.get_trainable_parameters()

        if not any(p.requires_grad for p in params):
            # MockGenerator: no actual gradient update possible
            return 0.0

        for _ in range(self.config.n_update_steps):
            self.optimizer.zero_grad()

            # --- PLACEHOLDER: replace with real LDM denoising loss ---
            # In Chemeleon2 integration:
            #   loss = ldm.compute_denoising_loss(structures)  # shape (N,)
            #   weighted_loss = -(advantages.detach() * loss).mean()
            #
            # GRPO clipping (PPO-style):
            #   ratio = exp(log_prob_new - log_prob_old)
            #   clipped = clip(ratio, 1-eps, 1+eps)
            #   loss = -min(ratio * adv, clipped * adv).mean()
            # ----------------------------------------------------------

            # Surrogate loss using fake parameter (for structural testing)
            fake_loss = sum(p.sum() for p in params if p.requires_grad)
            weighted = -(advantages.detach().mean() * fake_loss)
            loss = weighted + self.config.entropy_coeff * fake_loss.abs()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                params, self.config.gradient_clip
            )
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / self.config.n_update_steps

    def _extract_ehull_stats(self, structures) -> Dict[str, float]:
        """Try to extract E_hull stats from the last reward computation."""
        # EhullReward stores stats in structure properties if available
        ehull_values = []
        for s in structures:
            e = s.properties.get("energy_per_atom")
            if e is not None:
                ehull_values.append(e)

        if not ehull_values:
            return {}

        t = torch.tensor(ehull_values, dtype=torch.float32)
        return {
            "energy_pa_mean": t.mean().item(),
            "energy_pa_min": t.min().item(),
        }

    def _log_metrics(self, iteration: int, metrics: Dict[str, float]) -> None:
        msg = (
            f"[Iter {iteration:4d}] "
            f"reward={metrics['reward_mean']:.4f}±{metrics['reward_std']:.4f} "
            f"max={metrics['reward_max']:.4f} "
            f"loss={metrics['policy_loss']:.4f} "
            f"lr={metrics['lr']:.2e}"
        )
        if "energy_pa_mean" in metrics:
            msg += f" E_pa={metrics['energy_pa_mean']:.3f}"
        logger.info(msg)

        # Attempt W&B logging
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(metrics, step=iteration)
        except ImportError:
            pass

    def _save_checkpoint(self, tag) -> None:
        path = self.output_dir / f"generator_rl_{tag}.pt"
        self.generator.save_checkpoint(str(path))
        logger.info(f"Saved checkpoint: {path}")

        # Also save metrics history
        import json
        hist_path = self.output_dir / "history.json"
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)

    # ------------------------------------------------------------------
    # Convenience: evaluate without training
    # ------------------------------------------------------------------

    def evaluate(self, n: int = 100) -> Dict[str, Any]:
        """
        Sample n structures and compute metrics without updating the policy.

        Returns
        -------
        dict with keys: structures, rewards, reward_mean, reward_std,
                        stability_fraction (if EhullReward present)
        """
        structures = self.generator.sample(n=n, condition=self.condition)
        rewards = self.reward_fn(structures)

        result = {
            "structures": structures,
            "rewards": rewards,
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "n_structures": n,
        }

        # Check for EhullReward component
        from src.rewards.ehull_reward import EhullReward
        for comp in self.reward_fn.components:
            if isinstance(comp, EhullReward):
                result["stability_fraction"] = comp.stability_fraction(structures)
                break

        return result
