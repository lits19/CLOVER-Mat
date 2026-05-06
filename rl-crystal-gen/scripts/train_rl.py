#!/usr/bin/env python
"""
scripts/train_rl.py
--------------------
Entry point for RL-guided crystal structure generation.

Usage examples
--------------
# Use built-in Ag-Bi-I E_hull reward preset (mock generator for testing):
python scripts/train_rl.py --mock --elements Ag Bi I --n_iter 100

# Use Chemeleon2 LDM with MACE backend:
python scripts/train_rl.py \
    --ldm_checkpoint checkpoints/ldm_best.ckpt \
    --vae_checkpoint checkpoints/vae_best.ckpt \
    --reward_config configs/rewards/ag_bi_i_ehull.yaml \
    --n_iter 500 \
    --batch_size 64 \
    --device cuda

# Use custom reward YAML:
python scripts/train_rl.py \
    --mock \
    --reward_config configs/rewards/my_custom_reward.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure src is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rl.grpo_trainer import GRPOTrainer, GRPOConfig
from src.rl.reward_builder import build_ag_bi_i_ehull_reward, build_reward_from_yaml
from src.generator.chemeleon2_wrapper import MockGenerator, Chemeleon2Generator

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RL-guided crystal structure generation with E_hull reward"
    )

    # Generator options
    gen_g = p.add_argument_group("Generator")
    gen_g.add_argument(
        "--mock", action="store_true",
        help="Use MockGenerator (no GPU/Chemeleon2 required; for testing)"
    )
    gen_g.add_argument(
        "--ldm_checkpoint", type=str, default=None,
        help="Path to Chemeleon2 LDM checkpoint (.ckpt)"
    )
    gen_g.add_argument(
        "--vae_checkpoint", type=str, default=None,
        help="Path to Chemeleon2 VAE checkpoint (.ckpt)"
    )
    gen_g.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Compute device"
    )

    # Reward options
    rew_g = p.add_argument_group("Reward")
    rew_g.add_argument(
        "--reward_config", type=str, default=None,
        help="Path to reward YAML config. If not set, uses built-in Ag-Bi-I preset."
    )
    rew_g.add_argument(
        "--elements", nargs="+", default=["Ag", "Bi", "I"],
        help="Target elements for composition reward"
    )
    rew_g.add_argument(
        "--ehull_backend", type=str, default="mp_api",
        choices=["mp_api", "mace"],
        help="Backend for E_hull computation"
    )

    # Training options
    tr_g = p.add_argument_group("Training")
    tr_g.add_argument("--n_iter", type=int, default=200, help="Number of RL iterations")
    tr_g.add_argument("--batch_size", type=int, default=32, help="Structures per iteration")
    tr_g.add_argument("--group_size", type=int, default=8, help="GRPO group size")
    tr_g.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    tr_g.add_argument("--clip_eps", type=float, default=0.2, help="PPO clip epsilon")
    tr_g.add_argument("--kl_coeff", type=float, default=0.01, help="KL coefficient")
    tr_g.add_argument("--save_every", type=int, default=50, help="Save checkpoint interval")
    tr_g.add_argument("--log_every", type=int, default=10, help="Logging interval")
    tr_g.add_argument("--output_dir", type=str, default="outputs/rl_run", help="Output directory")
    tr_g.add_argument("--seed", type=int, default=42, help="Random seed")

    # W&B
    p.add_argument("--wandb_project", type=str, default=None, help="W&B project name")
    p.add_argument("--run_name", type=str, default=None, help="W&B run name")

    return p.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # --- Init W&B ---
    if args.wandb_project:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.run_name or f"rl_agbii_{args.ehull_backend}",
                config=vars(args),
            )
            logger.info(f"W&B initialized: {args.wandb_project}")
        except ImportError:
            logger.warning("wandb not installed; skipping W&B logging")

    # --- Build Generator ---
    if args.mock:
        logger.info("Using MockGenerator (testing mode)")
        generator = MockGenerator(seed=args.seed)
    elif args.ldm_checkpoint and args.vae_checkpoint:
        logger.info(f"Loading Chemeleon2 generator from {args.ldm_checkpoint}")
        generator = Chemeleon2Generator(
            checkpoint_path=args.ldm_checkpoint,
            vae_checkpoint_path=args.vae_checkpoint,
            device=args.device,
        )
    else:
        logger.error(
            "Specify either --mock OR both --ldm_checkpoint and --vae_checkpoint"
        )
        sys.exit(1)

    # --- Build Reward ---
    if args.reward_config:
        logger.info(f"Loading reward config from {args.reward_config}")
        reward_fn = build_reward_from_yaml(args.reward_config)
    else:
        logger.info(f"Using built-in Ag-Bi-I E_hull reward (backend={args.ehull_backend})")
        reward_fn = build_ag_bi_i_ehull_reward(ehull_backend=args.ehull_backend)

    # --- Build Trainer ---
    config = GRPOConfig(
        n_iterations=args.n_iter,
        batch_size=args.batch_size,
        group_size=args.group_size,
        lr=args.lr,
        clip_epsilon=args.clip_eps,
        kl_coeff=args.kl_coeff,
        save_every=args.save_every,
        log_every=args.log_every,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
    )

    condition = {"elements": args.elements}

    trainer = GRPOTrainer(
        generator=generator,
        reward_fn=reward_fn,
        config=config,
        condition=condition,
    )

    # --- Train ---
    logger.info("=" * 60)
    logger.info("Starting RL training for crystal structure generation")
    logger.info(f"  System: {'-'.join(args.elements)}")
    logger.info(f"  Reward backend: {args.ehull_backend}")
    logger.info(f"  Iterations: {args.n_iter}, batch_size: {args.batch_size}")
    logger.info("=" * 60)

    history = trainer.train()

    # --- Final Evaluation ---
    logger.info("\nRunning final evaluation (100 structures)...")
    eval_result = trainer.evaluate(n=100)
    logger.info(f"Final reward mean: {eval_result['reward_mean']:.4f}")
    if "stability_fraction" in eval_result:
        logger.info(
            f"Stability fraction (E_hull < 0.1 eV/atom): "
            f"{eval_result['stability_fraction']:.2%}"
        )

    # Save final structures
    import json
    from monty.serialization import dumpfn
    out_path = Path(args.output_dir) / "final_structures.json"
    try:
        dumpfn(eval_result["structures"], str(out_path))
        logger.info(f"Saved {len(eval_result['structures'])} structures to {out_path}")
    except Exception as e:
        logger.warning(f"Could not save structures: {e}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
