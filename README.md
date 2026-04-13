# Hydra-BitNet

**Cross-Layer Fused Speculative Decoding for Ternary-Weight LLMs**

*By Parrish Corcoran*

---

## Research Thesis

Modern speculative decoding (Medusa, EAGLE, etc.) taps the **final hidden
layer** of a frozen backbone to predict future tokens. This wastes two
opportunities specific to **bandwidth-bound ternary inference** (BitNet b1.58):

1. **Every intermediate layer's hidden state is computed for free** during
   the backbone forward pass, but only the last layer is used for speculation.
2. **BitNet's LUT-based GEMM kernels are weight-stream-bound**, meaning the
   verify batch costs approximately the same whether it contains 5 tokens or
   50. The speedup ceiling is determined by *acceptance rate*, not batch cost
   — fundamentally higher than on FP16/INT8 models where batch cost scales
   linearly.

Hydra-BitNet exploits both observations:

- **Cross-layer fusion:** instead of independent per-layer heads, a learned
  bottleneck fuses hidden states from 6+ backbone layers into a single
  representation that captures *both* shallow syntactic patterns (layers 4-8)
  and deep semantic context (layers 24-30). One fused predictor that can
  *route* across layers beats 30 independent predictors that each see only
  one layer.
- **Wide-shallow speculation trees** optimized for the bandwidth-bound cost
  model: saturate depth-1 acceptance (many candidates from the fused head),
  then depth-2, then stop. Depth-4 heads contribute <0.002 tokens/step in
  practice; those tree slots are better spent on more depth-1 candidates.
- **Adaptive speculation depth** driven by cross-layer agreement: when all
  layers agree on the next token, speculate deep (high acceptance expected);
  when layers disagree, speculate shallow or skip (avoid wasting the verify
  batch on certain failures).

### Theoretical ceiling

On standard GPU inference, speculative decoding is capped at ~2-3x because
the verify batch has linear cost in tokens. On BitNet's LUT kernels where
the MLP batch is free (weight-stream-bound) and only attention scales:

| Tree nodes | Attention overhead | Effective ceiling |
|---|---|---|
| 5 (current chain) | ~0% | 2.2x (measured) |
| 20 | ~5% | 3-3.5x |
| 50-100 | ~15-25% | **4-5x** |

The gap between 2.2x and 5x is the research target.

---

## Architecture

```
                        BitNet b1.58 Backbone (frozen, ternary LUT GEMM)
                        ═══════════════════════════════════════════════
  input ──► Layer 1 ──► Layer 2 ──► ... ──► Layer L ──► ... ──► Layer 30
                │           │                  │                    │
                ▼           ▼                  ▼                    ▼
              h_1         h_2                h_L                  h_30
                │           │                  │                    │
                └───────────┴──────────────────┴────────────────────┘
                                       │
                              ┌────────▼────────┐
                              │   Concat [6×H]  │   (tap layers 5,10,15,20,25,30)
                              │        │        │
                              │  W_fuse [6H→R]  │   R = 256 (shared bottleneck)
                              │        │        │
                              │     SiLU        │
                              │        │        │
                              │  W_pred [R→H]   │   (per-head, for k=1..K)
                              │        │        │
                              │  Shared LM head │   (tied tok_embd, H→V)
                              │   [H → 128256]  │
                              └────────┬────────┘
                                       │
                              logits for t+1, t+2, ...
                                       │
                              ┌────────▼────────┐
                              │  Speculation     │
                              │  Tree Builder    │
                              │  (wide-shallow)  │
                              └────────┬────────┘
                                       │
                              ┌────────▼────────┐
                              │  Verify Batch    │
                              │  (BitNet LUT     │
                              │   forward, ~free │
                              │   batch cost)    │
                              └────────┬────────┘
                                       │
                              ┌────────▼────────┐
                              │  Accept / Reject │
                              │  + KV Rollback   │
                              └─────────────────┘
```

### Fused Bottleneck Head (detail)

Each Hydra head fuses N hidden states through a shared low-rank bottleneck:

```
Inputs:  h_5, h_10, h_15, h_20, h_25, h_30   (each [B, T, H], H=2560)
Concat:  [B, T, 6H] = [B, T, 15360]
W_fuse:  [6H, R] = [15360, 256]                per-head, learned
Activation: SiLU
W_pred:  [R, H] = [256, 2560]                  per-head, learned
Residual: h_fused = h_30 + W_pred(SiLU(W_fuse(concat)))
LM head: logits = h_fused @ tok_embd.T          shared, frozen

Cost: ~41 MFLOPs per head = 2% of backbone decode
```

The residual connection from `h_30` means the head starts as the standard
Medusa-1 architecture (identity init on `W_pred`, zero on `W_fuse`) and
learns to incorporate earlier layers during training. This guarantees
Hydra is *at least* as good as single-layer Medusa from step 0.

### Optimal Head Allocation (data-driven)

Empirical acceptance rates from our Medusa baseline on BitNet b1.58 2B:

```
Head 1 (t+1 from layer 30):  67.6% acceptance  → 0.676 bonus tokens
Head 2 (t+2 from layer 30):  33.2% acceptance  → 0.224 bonus tokens
Head 3 (t+3 from layer 30):  14.2% acceptance  → 0.032 bonus tokens
Head 4 (t+4 from layer 30):   6.3% acceptance  → 0.002 bonus tokens
```

Head 4 contributes 0.002 tokens/step. A second depth-1 head from layer 16
(~50% accuracy) contributes 0.162 tokens/step — **80x more valuable.** The
optimal allocation saturates depth 1 first, then depth 2:

```
Priority 1: Depth-1 heads until p(accept at depth 1) > 0.95
Priority 2: Depth-2 heads until p(accept at depth 2) > 0.50
Priority 3: One depth-3 head if budget remains
Priority 4: Never depth-4 (not worth the tree slot)
```

---

## Relation to MedusaBitNet

This project builds on the proven
[MedusaBitNet](https://github.com/parrishcorcoran/MedusaBitNet) baseline:

| Component | MedusaBitNet (proven) | Hydra-BitNet (this repo) |
|---|---|---|
| Backbone | BitNet b1.58 2B | same |
| Head architecture | Single-layer residual, taps layer 30 only | Cross-layer fused bottleneck, taps 6 layers |
| Speculation topology | Linear chain (4 deep) | Wide-shallow tree (optimized for bandwidth-bound) |
| Training | Cached hidden states (layer 30 only) | Live backbone forward, tap all layers |
| Measured speedup | 2.2x | Target: 4-5x |
| C++ runtime | `llama-medusa` (chain driver) | `llama-hydra` (tree driver with fused heads) |

The MedusaBitNet C++ integration (model loader, graph build, decode
extraction, public API) carries over directly. Hydra adds:
- Multi-layer tap points in the graph build
- Fused bottleneck GEMM nodes
- Tree batch builder + adaptive depth
- Modified `llama_batch` filling with per-branch `seq_id` inheritance

---

## Project Status

| Phase | Status |
|---|---|
| Baseline Medusa on BitNet (2.2x proven) | ✅ Complete ([MedusaBitNet](https://github.com/parrishcorcoran/MedusaBitNet)) |
| Theoretical analysis (optimal head allocation) | ✅ Complete |
| Fused bottleneck head implementation (Python) | 🔧 In progress |
| Multi-layer caching / live-forward training | 🔧 In progress |
| C++ graph build for multi-layer taps | ⬜ Planned |
| Tree driver (`llama-hydra`) | ⬜ Planned |
| H2O KV cache integration | ⬜ Planned |
| Benchmark vs. vanilla bitnet.cpp | ⬜ Planned |

---

## Quick Start

*Full setup instructions coming. For now, see
[MedusaBitNet/docs/SETUP.md](https://github.com/parrishcorcoran/MedusaBitNet/blob/main/docs/SETUP.md)
for the baseline pipeline, which Hydra-BitNet extends.*

```bash
git clone https://github.com/parrishcorcoran/Hydra-BitNet.git
cd Hydra-BitNet
# setup instructions TBD
```

---

## Key Insight

> On bandwidth-bound ternary inference, the verify batch is nearly free.
> The ceiling for speculative decoding is set by acceptance rate alone —
> not by batch cost. Every technique that pushes acceptance rate higher
> translates directly to proportional speedup, with no diminishing returns
> from batch overhead. This makes BitNet the ideal substrate for aggressive
> speculative decoding: the architecture that benefits *least* from raw
> compute benefits *most* from smart speculation.

---

## Citation

```
@misc{corcoran2026hydra,
  title={Hydra-BitNet: Cross-Layer Fused Speculative Decoding
         for Ternary-Weight LLMs},
  author={Corcoran, Parrish},
  year={2026},
  url={https://github.com/parrishcorcoran/Hydra-BitNet}
}
```

## License

MIT
