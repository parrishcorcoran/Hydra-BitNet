"""
Hydra fused bottleneck heads for cross-layer speculative decoding.

Hydra-BitNet by Parrish Corcoran.

Each Hydra head takes hidden states from multiple backbone layers, fuses
them through a learned low-rank bottleneck, and projects to the vocab via
the shared (frozen) LM head. The fusion lets the head learn which layers
to trust per-context — shallow layers for syntactic predictions, deep
layers for semantic ones — rather than being limited to a single layer's
view like standard Medusa.

The residual connection from the deepest tapped layer (h_30) ensures the
head starts equivalent to a standard Medusa-1 head (identity init) and
can only improve from there.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HydraFusedHead(nn.Module):
    """
    One Hydra head: fuses N hidden states through a low-rank bottleneck.

    Architecture:
        concat([h_l1, h_l2, ..., h_lN])  →  W_fuse [N*H, R]  →  SiLU
            →  W_pred [R, H]  →  + h_deepest (residual)  →  LM head [H, V]

    The LM head weight is NOT owned by this module — it's passed in at
    forward time (tied to the backbone's token embedding).
    """

    def __init__(
        self,
        hidden_size: int,
        n_layers: int,
        bottleneck_dim: int = 256,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bottleneck_dim = bottleneck_dim

        self.w_fuse = nn.Linear(
            n_layers * hidden_size, bottleneck_dim, bias=False, dtype=dtype,
        )
        self.w_pred = nn.Linear(
            bottleneck_dim, hidden_size, bias=False, dtype=dtype,
        )
        # Zero-init w_pred so the head starts as identity (h_deepest passthrough).
        nn.init.zeros_(self.w_pred.weight)

    def forward(
        self,
        layer_hiddens: list[torch.Tensor],
        lm_head_weight: torch.Tensor,
        pos_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            layer_hiddens: list of N tensors, each [B, T, H]. Ordered
                shallow-to-deep (layer_hiddens[-1] is the deepest).
            lm_head_weight: [V, H] frozen vocab projection.
            pos_indices: optional [B, P] int64 for position subsampling.

        Returns:
            logits: [B, T_or_P, V]
        """
        # Residual base: deepest layer's hidden state.
        h_deep = layer_hiddens[-1]  # [B, T, H]

        # Concat all layers along the feature dim.
        fused_input = torch.cat(layer_hiddens, dim=-1)  # [B, T, N*H]

        # Bottleneck fusion.
        z = self.w_fuse(fused_input)  # [B, T, R]
        z = F.silu(z)
        z = self.w_pred(z)            # [B, T, H]

        # Residual: start as identity, learn to improve.
        h = h_deep + z               # [B, T, H]

        # Position subsampling before the expensive vocab projection.
        if pos_indices is not None:
            B, T, H = h.shape
            P = pos_indices.shape[1]
            idx = pos_indices.unsqueeze(-1).expand(-1, -1, H)  # [B, P, H]
            h = torch.gather(h, 1, idx)                        # [B, P, H]

        # Vocab projection via tied LM head.
        logits = F.linear(h, lm_head_weight)  # [B, T_or_P, V]
        return logits


class HydraHeads(nn.Module):
    """
    K Hydra heads, each predicting a different future token (t+1, t+2, ..., t+K).
    All heads share the same layer taps and LM head weight but have independent
    bottleneck parameters.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        n_fuse_layers: int,
        bottleneck_dim: int = 256,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([
            HydraFusedHead(hidden_size, n_fuse_layers, bottleneck_dim, dtype)
            for _ in range(num_heads)
        ])

    def forward(
        self,
        layer_hiddens: list[torch.Tensor],
        lm_head_weight: torch.Tensor,
        pos_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            layer_hiddens: list of N tensors, each [B, T, H].
            lm_head_weight: [V, H] frozen vocab projection.
            pos_indices: optional [B, P] int64.

        Returns:
            logits: [B, T_or_P, K, V] — one set of vocab logits per head.
        """
        head_logits = [
            head(layer_hiddens, lm_head_weight, pos_indices)
            for head in self.heads
        ]
        # Stack along head dimension: each is [B, T_or_P, V] → [B, T_or_P, K, V]
        return torch.stack(head_logits, dim=2)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


def hydra_loss(
    hydra_logits: torch.Tensor,
    targets: torch.Tensor,
    num_heads: int,
    pos_indices: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, list[float]]:
    """
    Cross-entropy across K Hydra heads + per-head top-1 accuracy.
    Same semantics as MedusaBitNet's medusa_loss — head i predicts token
    at position (t + i + 1).

    Args:
        hydra_logits: [B, T_or_P, K, V]
        targets: [B, T_full + 1]
        num_heads: K
        pos_indices: optional [B, P] int64 (same as passed to HydraHeads.forward)

    Returns:
        (loss, [acc_per_head])
    """
    V = hydra_logits.shape[-1]
    total_loss = torch.zeros((), dtype=hydra_logits.dtype, device=hydra_logits.device)
    accuracies: list[float] = []

    if pos_indices is None:
        B, T, K, _ = hydra_logits.shape
        T_plus_1 = targets.shape[1]
        assert T_plus_1 == T + 1

        for i in range(num_heads):
            shift = i + 1
            valid_len = T - shift + 1
            if valid_len <= 0:
                accuracies.append(0.0)
                continue
            logits_i = hydra_logits[:, :valid_len, i, :]
            targets_i = targets[:, shift : shift + valid_len]
            loss_i = F.cross_entropy(
                logits_i.reshape(-1, V).float(), targets_i.reshape(-1),
            )
            total_loss = total_loss + loss_i
            with torch.no_grad():
                acc = (logits_i.argmax(-1) == targets_i).float().mean().item()
                accuracies.append(acc)
    else:
        for i in range(num_heads):
            shift = i + 1
            logits_i = hydra_logits[:, :, i, :]
            target_idx = pos_indices + shift
            targets_i = torch.gather(targets, 1, target_idx)
            loss_i = F.cross_entropy(
                logits_i.reshape(-1, V).float(), targets_i.reshape(-1),
            )
            total_loss = total_loss + loss_i
            with torch.no_grad():
                acc = (logits_i.argmax(-1) == targets_i).float().mean().item()
                accuracies.append(acc)

    total_loss = total_loss / num_heads
    return total_loss, accuracies
