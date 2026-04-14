"""Hydra-BitNet ablation: single-layer bottleneck on cached hidden states.

Reuses MedusaBitNet's cached layer-30 hidden states (data/hidden.bin) to test
the Hydra bottleneck architecture against the standard Medusa head architecture
on identical data. This is the fast ablation path: prove the bottleneck head
is at least as good as standard Medusa before committing to multi-layer caching.

Single-layer Hydra (n_fuse_layers=1):
    h_30 --> W_fuse [H, R] --> SiLU --> W_pred [R, H] --> + h_30 (residual) --> LM head

vs standard Medusa:
    h_30 --> W_in [H, H] --> SiLU --> W_out [H, H] --> + h_30 (residual) --> LM head

Same number of matmuls, but Hydra has a low-rank bottleneck (R=256 vs full H=2560).
If Hydra matches Medusa at matched compute, the architecture is validated for
scaling to multi-layer fusion.
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import math
import time
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from hydra_head import HydraHeads, hydra_loss

# Reuse MedusaBitNet's dataset utility
MEDUSA_PATH = "/home/cpinchington/MedusaBitNet"
sys.path.insert(0, MEDUSA_PATH)
from dataset import PackedTokenDataset


class CachedSingleLayerDataset(Dataset):
    """Layer-30 hidden states + token targets from MedusaBitNet cache."""
    def __init__(self, hidden_path, token_bin_path, seq_len, hidden_size):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self._hidden = np.memmap(hidden_path, dtype=np.uint16, mode="r")
        per_seq = seq_len * hidden_size
        assert self._hidden.size % per_seq == 0
        self.num_samples = self._hidden.size // per_seq
        self._tokens = PackedTokenDataset(token_bin_path, seq_len)

    def __len__(self):
        return min(self.num_samples, len(self._tokens))

    def __getitem__(self, idx):
        per_seq = self.seq_len * self.hidden_size
        start = idx * per_seq
        flat = np.asarray(self._hidden[start:start + per_seq])
        hidden = torch.from_numpy(flat).view(torch.bfloat16).view(self.seq_len, self.hidden_size)
        targets = self._tokens[idx]
        return hidden, targets


def collate(batch):
    hiddens = torch.stack([b[0] for b in batch])
    targets = torch.stack([b[1] for b in batch])
    return hiddens, targets


def warmup_cosine_lr(step, warmup, total, base_lr):
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * min(1.0, progress)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden_path", default=f"{MEDUSA_PATH}/data/hidden.bin")
    p.add_argument("--bin_path", default=f"{MEDUSA_PATH}/data/tokens.bin")
    p.add_argument("--lm_head_path", default=f"{MEDUSA_PATH}/data/lm_head.pt")
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--hidden_size", type=int, default=2560)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum_steps", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)

    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--bottleneck_dim", type=int, default=256)

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--loss_positions", type=int, default=256)

    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--ckpt_dir", default="checkpoints_ablation")
    p.add_argument("--ckpt_every", type=int, default=500)
    p.add_argument("--label", default="hydra-cached")
    args = p.parse_args()

    torch.manual_seed(0)
    device = torch.device("cpu")
    print(f"[hydra-ablation] device = {device}")
    print(f"[hydra-ablation] num_heads = {args.num_heads}, bottleneck = {args.bottleneck_dim}")
    print(f"[hydra-ablation] data: {args.hidden_path}")

    # Pre-transpose lm_head for einsum-friendly shape [H, V]
    lm_head = torch.load(args.lm_head_path, map_location="cpu", weights_only=True).to(torch.bfloat16).contiguous()
    # Keep as [V, H] — HydraFusedHead uses F.linear which takes weight as [V, H]
    print(f"[hydra-ablation] lm_head shape: {lm_head.shape}")

    dataset = CachedSingleLayerDataset(
        args.hidden_path, args.bin_path, args.seq_len, args.hidden_size,
    )
    print(f"[hydra-ablation] dataset: {len(dataset)} sequences")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate,
        pin_memory=False, drop_last=True,
    )

    heads = HydraHeads(
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        n_fuse_layers=1,  # Single-layer ablation
        bottleneck_dim=args.bottleneck_dim,
        dtype=torch.bfloat16,
    )
    heads.train()
    print(f"[hydra-ablation] head params: {sum(p.numel() for p in heads.parameters()):,}")

    optim = torch.optim.AdamW(
        heads.trainable_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    os.makedirs(args.ckpt_dir, exist_ok=True)
    step = 0
    accum = 0
    t0 = time.time()
    loss_accum = 0.0
    optim.zero_grad(set_to_none=True)
    data_iter = iter(loader)

    results_log = []

    while step < args.max_steps:
        try:
            hidden, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            hidden, targets = next(data_iter)

        B, T, H = hidden.shape
        max_valid = T - args.num_heads
        if args.loss_positions > 0 and args.loss_positions < max_valid:
            P = args.loss_positions
            perm = torch.randperm(max_valid)[:P]
            pos_indices = perm.sort().values.unsqueeze(0).expand(B, -1).contiguous()
        else:
            pos_indices = None

        # Single-layer: pass [h_30] as the only layer
        with torch.cpu.amp.autocast(dtype=torch.bfloat16):
            logits = heads([hidden], lm_head, pos_indices=pos_indices)
            loss, head_accs = hydra_loss(
                logits, targets, args.num_heads, pos_indices=pos_indices,
            )
            loss = loss / args.grad_accum_steps

        loss.backward()
        loss_accum += loss.item() * args.grad_accum_steps
        accum += 1

        if accum < args.grad_accum_steps:
            continue

        lr_now = warmup_cosine_lr(step, args.warmup_steps, args.max_steps, args.lr)
        for g in optim.param_groups:
            g["lr"] = lr_now

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(heads.trainable_parameters(), args.grad_clip)
        optim.step()
        optim.zero_grad(set_to_none=True)

        step += 1
        accum = 0

        if step % args.log_every == 0:
            dt = time.time() - t0
            avg_loss = loss_accum / (args.log_every * args.grad_accum_steps)
            head_acc_str = " ".join(f"acc@{i+1}={a:.3f}" for i, a in enumerate(head_accs))
            msg = f"step {step:4d} | loss {avg_loss:.4f} | lr {lr_now:.2e} | {dt/args.log_every:.2f}s/step | {head_acc_str}"
            print(msg, flush=True)
            results_log.append({
                "step": step, "loss": avg_loss, "lr": lr_now,
                "head_accs": head_accs, "seconds_per_step": dt/args.log_every,
            })
            loss_accum = 0.0
            t0 = time.time()

        if step % args.ckpt_every == 0 or step == args.max_steps:
            ckpt_path = os.path.join(args.ckpt_dir, f"{args.label}_step{step}.pt")
            torch.save({
                "heads": heads.state_dict(),
                "step": step,
                "num_heads": args.num_heads,
                "bottleneck_dim": args.bottleneck_dim,
                "n_fuse_layers": 1,
                "hidden_size": args.hidden_size,
                "results_log": results_log,
                "args": vars(args),
            }, ckpt_path)
            print(f"[hydra-ablation] saved {ckpt_path}")

    print("[hydra-ablation] done")


if __name__ == "__main__":
    main()
