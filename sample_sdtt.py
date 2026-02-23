#!/usr/bin/env python3
"""
SDTT Sampling Script — Fair Comparison with D-Perflow

Aligned parameters:
  - num_steps=16      <-> D-Perflow --num_time_windows 16
  - num_samples=1024  <-> D-Perflow --num_samples 1024
  - seq_len=1024      (MDLM/SDTT default context window)
  - batch_size=16     <-> D-Perflow --batch_size 16
  - model: dit-orig-small (Small ~64-100M params)

Usage examples:
  # Load from local directory (config.json + model.safetensors):
  python sample_sdtt.py --model_dir /path/to/kld_r7

  # Load from HuggingFace:
  python sample_sdtt.py --from_hf --loss kld --round 7

  # Custom parameters:
  python sample_sdtt.py --model_dir /path/to/kld_r7 \
      --num_steps 16 --num_samples 1024 --batch_size 16 --seq_len 1024
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import lightning as L
from tqdm import trange
from loguru import logger
from omegaconf import OmegaConf
from safetensors.torch import load_file
from transformers import AutoTokenizer

# Ensure sdtt package is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sdtt.core.distill.multi_round_sdtt import (
    MultiRoundSDTT,
    load_small_student,
)


def load_from_local(model_dir):
    """Load a MultiRoundSDTT model from a local directory.

    Expects the directory to contain:
      - config.json        (OmegaConf config saved by push_to_hub)
      - model.safetensors  (model state_dict)

    This mirrors MultiRoundSDTT.from_pretrained but reads from
    disk instead of HuggingFace Hub.
    """
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    ckpt_path = model_dir / "model.safetensors"

    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found at {config_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"model.safetensors not found at {ckpt_path}")

    logger.info(f"Loading config from {config_path}")
    config = OmegaConf.load(config_path)
    ckpt = load_file(str(ckpt_path))
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.name)

    # MultiRoundSDTT.__init__ calls prepare_teacher_and_student(),
    # which may download the MDLM teacher from HuggingFace (cached
    # after the first run). The teacher is NOT used for sampling,
    # only for distillation training.
    model = MultiRoundSDTT(config, tokenizer, verbose=False)
    model.load_state_dict(ckpt)
    return model


def run_sampling(model, num_steps, num_samples, seq_len, batch_size, device):
    """Run unconditional sampling and return numpy array of token ids."""
    model = model.to(device)
    model.eval()

    # Use EMA weights if available (standard for evaluation)
    use_ema = model.ema is not None
    if use_ema:
        model.store_ema()
        logger.info("Using EMA weights for sampling")

    assert num_samples % batch_size == 0, (
        f"num_samples ({num_samples}) must be divisible by "
        f"batch_size ({batch_size})"
    )
    n_rounds = num_samples // batch_size

    logger.info(
        f"Sampling config: num_steps={num_steps}, num_samples={num_samples}, "
        f"seq_len={seq_len}, batch_size={batch_size}"
    )

    all_samples = []
    for _ in trange(n_rounds, desc=f"Sampling (steps={num_steps})"):
        with torch.no_grad(), torch.amp.autocast("cuda"):
            out = model.sample(
                n_samples=batch_size,
                num_steps=num_steps,
                seq_len=seq_len,
                sampler="ancestral",
                cache_preds=False,
                verbose=False,
                add_bos=False,
                add_eos=False,
            )
        all_samples.append(out.cpu())

    all_samples = torch.cat(all_samples, dim=0).numpy()
    all_samples = all_samples[:num_samples]

    # Restore original weights
    if use_ema:
        model.restore_ema()

    return all_samples


def main():
    parser = argparse.ArgumentParser(
        description="SDTT Unconditional Sampling (D-Perflow aligned)"
    )

    # --- Model source (mutually exclusive) ---
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--model_dir",
        type=str,
        help="Path to local model directory (config.json + model.safetensors)",
    )
    source.add_argument(
        "--from_hf",
        action="store_true",
        help="Load model from HuggingFace (jdeschena/sdtt)",
    )

    # HuggingFace options
    parser.add_argument(
        "--loss", type=str, default="kld",
        choices=["kld", "mse", "tvd"],
        help="Distillation loss type (only with --from_hf). Default: kld",
    )
    parser.add_argument(
        "--round", type=int, default=7,
        help="Distillation round 1-7 (only with --from_hf). Default: 7",
    )

    # --- Sampling parameters (defaults aligned with D-Perflow) ---
    parser.add_argument(
        "--num_steps", type=int, default=16,
        help="Number of sampling steps (= D-Perflow --num_time_windows). Default: 16",
    )
    parser.add_argument(
        "--num_samples", type=int, default=1024,
        help="Total number of samples to generate. Default: 1024",
    )
    parser.add_argument(
        "--seq_len", type=int, default=1024,
        help="Sequence length (tokens). Default: 1024",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size per forward pass. Default: 16",
    )

    # --- Output / misc ---
    parser.add_argument(
        "--output_dir", type=str, default="./samples_output",
        help="Directory to save .npz samples. Default: ./samples_output",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    # Reproducibility
    L.seed_everything(args.seed)

    # ---- Load model ----
    if args.from_hf:
        logger.info(
            f"Loading from HuggingFace: loss={args.loss}, round={args.round}"
        )
        model = load_small_student(loss=args.loss, round=args.round)
    else:
        logger.info(f"Loading from local directory: {args.model_dir}")
        model = load_from_local(args.model_dir)

    # ---- Sample ----
    all_samples = run_sampling(
        model,
        num_steps=args.num_steps,
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=args.device,
    )

    # ---- Save ----
    output_dir = Path(args.output_dir)
    # Use same subdirectory structure as the built-in sample_uncond
    save_dir = output_dir / "uncond"
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = save_dir / (
        f"uncond_steps{args.num_steps}"
        f"_n{args.num_samples}"
        f"_seq{args.seq_len}"
        f"_{timestamp}.npz"
    )

    metadata = dict(
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        sampler="ancestral",
        from_ema=True,
        add_bos=False,
        add_eos=False,
    )

    np.savez(save_path, samples=all_samples, metadata=metadata)
    logger.info(f"Saved {len(all_samples)} samples -> {save_path}")
    logger.info(f"Sample shape: {all_samples.shape}")


if __name__ == "__main__":
    main()
