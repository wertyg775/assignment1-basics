#!/usr/bin/env python3
import argparse
import math
import os
import time
import numpy as np
import torch
from pathlib import Path
import wandb

from cs336_basics.nn import cross_entropy
from cs336_basics.transformers import TransformerLM
from cs336_basics.optimizer import AdamW, cosine_scheduling, clip_grad_norm  
from cs336_basics.train_utils import get_batch, save_checkpoint, load_checkpoint

import os

DATASET_PATHS = {
    "tinystories": {
        "train": os.path.join("data", "TinyStoriesV2-GPT4-train.txt"),
        "valid": os.path.join("data", "TinyStoriesV2-GPT4-valid.txt"),
    },
    "openweb": {
        "train": os.path.join("data", "owt_train.txt"),
        "valid": os.path.join("data", "owt_valid.txt"),
    },
}

@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    data: np.ndarray,
    eval_iters: int,
    batch_size: int,
    context_length: int,
    device: str,
) -> float:
    """Evaluate average cross-entropy loss over a fixed number of batches."""
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = get_batch(data, batch_size, context_length, device)
        logits = model(x)  # (B, T, vocab_size)
        loss = cross_entropy(logits, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CS336 Transformer Language Model"
    )

    # for real training run
    parser.add_argument(
        "--execute",
        help="Command flag for live training run"
    )

    # Data
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        choices=["tinystories", "openweb"],
        help="Select dataset to train on",
    )
    
    parser.add_argument(
        "--data_dtype",
        type=str,
        default="uint16",
        help="NumPy dtype of token array on disk",
    )


    # Model Architecture
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--theta", type=float, default=10000.0)

    # Training & Optimization
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--min_lr", type=float, default=3e-4)
    parser.add_argument("--warmup_iters", type=int, default=200)
    parser.add_argument("--cosine_cycle_iters", type=int, default=5000)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # System & Logging
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--eval_iters", type=int, default=20)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint file to resume from",
    )
    return parser.parse_args()



def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_path = DATASET_PATHS[args.data]["train"]
    val_path = DATASET_PATHS[args.data]["valid"]

    # Device configuration
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"Running training on device: {device}")

    # Memory-mapped dataset loading
    train_data = np.memmap(train_path, dtype=args.data_dtype, mode="r")
    val_data = np.memmap(val_path, dtype=args.data_dtype, mode="r")

    # Instantiate TransformerLM
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_heads=args.num_heads,
        theta=args.theta,
        device=device,
        dtype=torch.float32,
    )
    model.to(device)

    # Configure optimizer with weight decay exclusion for 1D params (biases/norms)
    decay_params = [
        p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2
    ]
    nodecay_params = [
        p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2
    ]
    optim_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = AdamW(
        optim_groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8
    )

    # Resume from checkpoint if specified
    start_iter = 0
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
        print(f"Resumed at iteration: {start_iter}")

    # Training Loop
    model.train()
    t0 = time.time()

    for iteration in range(start_iter, args.max_iters):
        # Update learning rate according to cosine schedule
        lr = cosine_scheduling(
            iteration,
            args.lr,
            args.min_lr,
            args.warmup_iters,
            args.cosine_cycle_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Fetch batch
        x, y = get_batch(
            train_data, args.batch_size, args.context_length, device
        )

        # Forward pass
        logits = model(x)  # shape: (B, T, vocab_size)
        loss = cross_entropy(logits, y)

        # Backward pass & optimization step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if args.grad_clip > 0.0:
            clip_grad_norm(model.parameters(), args.grad_clip)

        optimizer.step()

        # Logging
        if (iteration + 1) % args.log_interval == 0 or iteration == start_iter:
            dt = time.time() - t0
            toks_per_sec = (args.batch_size * args.context_length * args.log_interval) / dt if iteration != start_iter else 0
            print(
                f"Iter {iteration + 1}/{args.max_iters} | "
                f"Loss: {loss.item():.4f} | "
                f"LR: {lr:.2e} | "
                f"Speed: {toks_per_sec:.0f} tokens/s"
            )
            t0 = time.time()

        # Validation & Checkpointing
        if (iteration + 1) % args.eval_interval == 0:
            if val_data is not None:
                val_loss = estimate_loss(
                    model,
                    val_data,
                    args.eval_iters,
                    args.batch_size,
                    args.context_length,
                    device,
                )
                print(f"--> [Eval] Iter {iteration + 1} | Val Loss: {val_loss:.4f}")

            ckpt_path = os.path.join(
                args.checkpoint_dir, f"ckpt_{iteration + 1}.pt"
            )
            save_checkpoint(model, optimizer, iteration + 1, ckpt_path)
            print(f"--> Checkpoint saved to {ckpt_path}")

    # Final checkpoint
    final_path = os.path.join(args.checkpoint_dir, "ckpt_final.pt")
    save_checkpoint(model, optimizer, args.max_iters, final_path)
    print(f"Training complete. Final checkpoint saved to {final_path}")


if __name__ == "__main__":
    main()