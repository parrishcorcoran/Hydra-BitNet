"""
Hydra-BitNet training: cross-layer fused speculative heads on frozen BitNet.

Hydra-BitNet by Parrish Corcoran.

Unlike MedusaBitNet (which caches layer-30 hidden states to disk), Hydra
needs hidden states from MULTIPLE layers. On fast hardware (Strix Halo
iGPU, any discrete GPU) it's simpler and faster to run the backbone live
per training step and tap intermediate layers in real-time, rather than
caching 6×20 GB to disk.

On CPU-only hardware, a multi-layer caching variant can be added later
if needed. For now this script assumes a device fast enough that the
frozen backbone forward is not the bottleneck.

Usage:
    python train_hydra.py \
        --backbone microsoft/bitnet-b1.58-2B-4T \
        --tap_layers 5 10 15 20 25 30 \
        --num_heads 2 \
        --bottleneck_dim 256 \
        --max_steps 2000 \
        --device cuda
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")

import argparse
import math
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hydra_head import HydraHeads, hydra_loss

# Reuse MedusaBitNet's dataset utilities if available, else provide fallback.
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "MedusaBitNet"))
try:
    from dataset import PackedTokenDataset, PackingConfig, build_token_bin, collate_packed
except ImportError:
    raise ImportError(
        "Could not import dataset utilities. Make sure MedusaBitNet/ is a "
        "sibling directory, or copy dataset.py into this repo."
    )


def parse_args():
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--dataset_name", default="tatsu-lab/alpaca")
    p.add_argument("--dataset_split", default="train")
    p.add_argument("--bin_path", default="data/tokens.bin")
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum_steps", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)

    # Model
    p.add_argument("--backbone", default="microsoft/bitnet-b1.58-2B-4T")
    p.add_argument("--tap_layers", type=int, nargs="+", default=[5, 10, 15, 20, 25, 30],
                   help="1-indexed layer numbers to tap hidden states from")
    p.add_argument("--num_heads", type=int, default=2,
                   help="Hydra heads (depth of speculation: t+1..t+K)")
    p.add_argument("--bottleneck_dim", type=int, default=256)

    # Optim
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--loss_positions", type=int, default=256,
                   help="Random position subsampling in loss (0=full seq)")

    # Logging / ckpt
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--ckpt_every", type=int, default=500)

    # Device
    p.add_argument("--device", default="cpu",
                   help="cpu, cuda, or cuda:N")

    return p.parse_args()


def warmup_cosine_lr(step, warmup, total, base_lr):
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * min(1.0, progress)))


def main():
    args = parse_args()
    torch.manual_seed(0)
    device = torch.device(args.device)
    print(f"[hydra] device = {device}")
    print(f"[hydra] tap_layers = {args.tap_layers}")
    print(f"[hydra] num_heads = {args.num_heads}, bottleneck_dim = {args.bottleneck_dim}")

    # ---- Data ---------------------------------------------------------------
    pack_cfg = PackingConfig(
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        seq_len=args.seq_len,
        bin_path=args.bin_path,
        tokenizer_name_or_path=args.backbone,
    )
    build_token_bin(pack_cfg)
    dataset = PackedTokenDataset(args.bin_path, args.seq_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_packed,
        pin_memory=(device.type != "cpu"),
        drop_last=True,
    )

    # ---- Backbone (frozen) --------------------------------------------------
    from transformers import AutoModelForCausalLM

    print(f"[hydra] loading backbone {args.backbone}")
    backbone = AutoModelForCausalLM.from_pretrained(
        args.backbone, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.to(device)

    hidden_size = backbone.config.hidden_size
    n_backbone_layers = backbone.config.num_hidden_layers

    # Validate tap layers.
    for tl in args.tap_layers:
        assert 1 <= tl <= n_backbone_layers, (
            f"tap_layer {tl} out of range [1, {n_backbone_layers}]"
        )

    # Grab the tied LM head weight.
    lm_head_weight = backbone.get_output_embeddings().weight.detach().to(torch.bfloat16)
    # Keep on device as a buffer.
    lm_head_weight = lm_head_weight.to(device)

    # ---- Hydra heads --------------------------------------------------------
    heads = HydraHeads(
        hidden_size=hidden_size,
        num_heads=args.num_heads,
        n_fuse_layers=len(args.tap_layers),
        bottleneck_dim=args.bottleneck_dim,
        dtype=torch.bfloat16,
    )
    heads.train()
    heads.to(device)
    print(f"[hydra] head params: {sum(p.numel() for p in heads.parameters()):,}")

    # ---- Optimizer ----------------------------------------------------------
    optim = torch.optim.AdamW(
        heads.trainable_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # ---- Training loop ------------------------------------------------------
    os.makedirs(args.ckpt_dir, exist_ok=True)
    step = 0
    accum = 0
    t0 = time.time()
    loss_accum = 0.0
    optim.zero_grad(set_to_none=True)
    data_iter = iter(loader)

    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        # batch: [B, seq_len+1] int64
        inputs = batch[:, :-1].contiguous().to(device, non_blocking=True)
        targets = batch.to(device, non_blocking=True)

        # ---- Frozen backbone forward, tap intermediate layers ----
        with torch.no_grad():
            outputs = backbone(
                input_ids=inputs,
                output_hidden_states=True,
                use_cache=False,
            )
            # outputs.hidden_states is a tuple of (n_layers+1) tensors:
            # index 0 = embedding output, index i = output of layer i.
            all_hidden = outputs.hidden_states
            layer_hiddens = [all_hidden[tl].detach() for tl in args.tap_layers]

        # ---- Position subsampling ----
        B, T, H = layer_hiddens[0].shape
        max_valid = T - args.num_heads
        if args.loss_positions > 0 and args.loss_positions < max_valid:
            P = args.loss_positions
            perm = torch.randperm(max_valid, device=device)[:P]
            pos_indices = perm.sort().values.unsqueeze(0).expand(B, -1).contiguous()
        else:
            pos_indices = None

        # ---- Hydra forward + loss ----
        hydra_logits = heads(layer_hiddens, lm_head_weight, pos_indices=pos_indices)
        loss, head_accs = hydra_loss(
            hydra_logits, targets, args.num_heads, pos_indices=pos_indices,
        )
        loss = loss / args.grad_accum_steps

        loss.backward()
        loss_accum += loss.item() * args.grad_accum_steps
        accum += 1

        if accum < args.grad_accum_steps:
            continue

        # ---- Optimizer step ----
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
            msg = (
                f"step {step:5d} | loss {avg_loss:.4f} | lr {lr_now:.2e} | "
                f"{dt/args.log_every:.2f}s/step | "
                + " ".join(f"acc@{i+1}={a:.3f}" for i, a in enumerate(head_accs))
            )
            print(msg, flush=True)
            loss_accum = 0.0
            t0 = time.time()

        if step % args.ckpt_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"hydra_heads_step{step}.pt")
            torch.save({
                "heads": heads.state_dict(),
                "step": step,
                "tap_layers": args.tap_layers,
                "num_heads": args.num_heads,
                "bottleneck_dim": args.bottleneck_dim,
                "hidden_size": hidden_size,
            }, ckpt_path)
            print(f"[hydra] saved {ckpt_path}")

    print("[hydra] done")


if __name__ == "__main__":
    main()
