# Acknowledgments

Hydra-BitNet is built on the work of many researchers, engineers, and open-source
contributors. This project would not exist without their foundational contributions.

## Core Technologies

**BitNet & BitNet b1.58**
Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Huaijie Wang, Lingxiao Ma,
Fan Yang, Ruiping Wang, Yi Wu, Furu Wei — Microsoft Research.
The ternary-weight architecture and the insight that 1.58-bit models can match
full-precision quality at dramatically lower compute cost. Their work made
efficient LLM inference on commodity CPUs a reality.

**bitnet.cpp**
Microsoft and Eddie Wang — the optimized C++ inference engine with lookup-table
(LUT) GEMM kernels for ternary weights. The bandwidth-bound cost model of these
kernels is what makes Hydra's wide-shallow speculation trees viable: the verify
batch is nearly free, so the speedup ceiling is determined by acceptance rate
alone.

**Medusa**
Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen,
Tri Dao — "Medusa: Simple LLM Inference Acceleration Framework with Multiple
Decoding Heads" (2024). The tree-structured speculative decoding framework and
the key insight that multiple lightweight heads on a frozen backbone can
self-speculate without a separate draft model. Hydra's architecture is a direct
descendant of Medusa's.

**llama.cpp**
Georgi Gerganov and the hundreds of contributors to the llama.cpp project.
The entire C++ inference stack that bitnet.cpp builds on — GGUF format, ggml
tensor library, KV cache, attention implementation, batched decoding,
speculative decoding primitives, and the examples that served as our reference
implementation. An extraordinary piece of open-source engineering.

## Speculative Decoding Foundations

**Speculative Decoding**
Yaniv Leviathan, Matan Kalman, Yossi Matias — Google Research. "Fast Inference
from Transformers via Speculative Decoding" (2023). The original framework for
draft-then-verify token generation.

Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau,
Laurent Sifre, John Jumper — DeepMind. "Accelerating Large Language Model
Decoding with Speculative Sampling" (2023). Independent concurrent work on the
same core idea.

**EAGLE**
Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang — "EAGLE: Speculative
Sampling Requires Rethinking Feature Uncertainty" (2024). Demonstrated that
combining the last layer's features with token embeddings improves draft quality.

**LayerSkip**
Mostafa Elhoushi, Akshat Shrivastava, Diana Liskovich, Basil Hosmer, Bram
Wasti, Liangzhen Lai, Anas Mahmoud, Bilge Acun, Saurabh Agarwal, Ahmed Roman,
Ahmed A. Aly, Beidi Chen, Carole-Jean Wu — Meta. "LayerSkip: Enabling
Early-Exit Inference and Self-Speculative Decoding" (2024). Showed that
intermediate layers can serve as draft predictors in a single model.

## KV Cache Efficiency

**H2O**
Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai,
Zhao Song, Yuandong Tian, Zhangyang Wang, Beidi Chen — "H2O: Heavy-Hitter
Oracle for Efficient Generative Inference of Large Language Models" (2023).
The attention-score-based KV cache eviction policy that we plan to integrate
with Hydra's speculative decoding pipeline.

## Frameworks & Tools

**PyTorch**
Meta AI and the PyTorch contributors. The training framework used for all head
training in both MedusaBitNet and Hydra-BitNet.

**Hugging Face Transformers**
Thomas Wolf, Lysandre Debut, Julien Chaumond, and the Hugging Face team.
Model loading, tokenization, and the pretrained BitNet b1.58 checkpoints that
made this work possible without training a backbone from scratch.

**Intel Extension for PyTorch (IPEX)**
Intel Corporation. AVX-512 fusion and bf16 optimization on the HP Z8 G4
workstation where the initial MedusaBitNet training and validation was done.

## Data

**Alpaca**
Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos
Guestrin, Percy Liang, Tatsunori B. Hashimoto — Stanford. The instruction-
tuning dataset used for training the Medusa and Hydra heads.

## Hardware

**HP Z8 G4 Workstation**
The dual-socket Xeon Platinum box where the initial MedusaBitNet pipeline was
developed, validated, and the 2.2x baseline was established. 11.5 hours of
hidden-state caching, 2 days of head training, and countless debug cycles.

**GMKtec EVO X2 (AMD Strix Halo)**
The consumer mini-PC where the Medusa baseline was benchmarked at 2.2x and
where Hydra-BitNet development continues. Its RDNA 3.5 iGPU and unified
LPDDR5x memory make it an ideal testbed for bandwidth-bound ternary inference
research.

## AI Assistance

This project was developed with substantial assistance from **Claude**
(Anthropic), which helped with architecture design, C++ implementation,
debugging, performance analysis, and the mathematical framework for optimal
head allocation. Claude's contributions are marked in commit messages with
`Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>`.

## Personal

Thank you to everyone in the open-source AI community who publishes their
code, models, and research openly. This project combines ideas from at least
six different research groups across four companies and two universities. None
of it would be possible without the culture of open publication and open-source
tooling that this community has built.

— Parrish Corcoran, April 2026
