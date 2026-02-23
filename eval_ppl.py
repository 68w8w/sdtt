#!/usr/bin/env python3
"""
Compute AR Perplexity of generated samples using GPT-2-large.

Reads .npz files produced by sample_sdtt.py and evaluates
autoregressive perplexity — the standard metric used in
MDLM / SDTT / D-Perflow papers.

Usage:
  # Evaluate a single .npz file:
  python eval_ppl.py --npz_path ./samples_output/uncond/uncond_steps16_n1024_seq1024_*.npz

  # Evaluate all .npz files under a directory:
  python eval_ppl.py --samples_dir ./samples_output/uncond

  # Use a different AR model:
  python eval_ppl.py --samples_dir ./samples_output/uncond --ar_model gpt2-large

  # Adjust batch size (lower if OOM):
  python eval_ppl.py --samples_dir ./samples_output/uncond --batch_size 8
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from tqdm import trange
from transformers import AutoModelForCausalLM


def compute_ppl(samples: np.ndarray, ar_model, device, batch_size=16):
    """Compute autoregressive perplexity.

    Args:
        samples: int array of shape (N, seq_len), token ids
        ar_model: causal LM used for evaluation
        device: torch device
        batch_size: evaluation batch size

    Returns:
        ppl: float, perplexity (exp of mean NLL)
        all_losses: list of per-sample NLL values
    """
    n_samples = samples.shape[0]
    all_losses = []

    for idx in trange(0, n_samples, batch_size, desc="Computing AR PPL"):
        batch = torch.tensor(
            samples[idx : idx + batch_size], dtype=torch.long
        ).to(device)

        with torch.no_grad():
            logits = ar_model(batch).logits[:, :-1]  # (B, L-1, V)

        log_probs = torch.log_softmax(logits, dim=-1)
        # Gather the log-prob of the actual next token
        targets = batch[:, 1:]  # (B, L-1)
        nll = -torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1))
        nll = nll.squeeze(-1)  # (B, L-1)

        per_sample_nll = nll.mean(dim=-1)  # average over positions
        all_losses.extend(per_sample_nll.cpu().tolist())

    avg_nll = np.mean(all_losses)
    ppl = float(np.exp(avg_nll))
    return ppl, all_losses


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate AR perplexity of SDTT samples"
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--npz_path", type=str,
        help="Path to a single .npz file",
    )
    source.add_argument(
        "--samples_dir", type=str,
        help="Directory containing .npz files (searches recursively)",
    )

    parser.add_argument(
        "--ar_model", type=str, default="gpt2-large",
        help="HuggingFace AR model for PPL evaluation. Default: gpt2-large",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for AR evaluation. Default: 16",
    )
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Collect .npz files
    if args.npz_path:
        npz_files = [Path(args.npz_path)]
    else:
        npz_files = sorted(Path(args.samples_dir).rglob("*.npz"))

    if not npz_files:
        logger.error("No .npz files found.")
        return

    logger.info(f"Found {len(npz_files)} .npz file(s)")

    # Load AR model once
    logger.info(f"Loading AR model: {args.ar_model}")
    ar_model = AutoModelForCausalLM.from_pretrained(args.ar_model).eval()
    ar_model = ar_model.to(args.device)
    logger.info("AR model loaded")

    # Evaluate each file
    for npz_path in npz_files:
        logger.info(f"Evaluating: {npz_path}")
        data = np.load(npz_path, allow_pickle=True)
        samples = data["samples"]

        metadata = {}
        if "metadata" in data:
            metadata = data["metadata"].item()

        num_steps = metadata.get("num_steps", "?")
        seq_len = metadata.get("seq_len", "?")
        n = samples.shape[0]

        logger.info(
            f"  samples: {n}, seq_len: {seq_len}, num_steps: {num_steps}"
        )

        ppl, all_losses = compute_ppl(
            samples, ar_model, args.device, args.batch_size
        )

        avg_nll = np.mean(all_losses)
        std_nll = np.std(all_losses)

        logger.info(f"  AR PPL = {ppl:.2f}  (avg NLL = {avg_nll:.4f} +/- {std_nll:.4f})")

    logger.info("Done.")


if __name__ == "__main__":
    main()
