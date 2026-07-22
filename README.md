# minigpt

GPT-2 built from scratch in PyTorch. Includes pretrained weight loading, KV cache, and speculative decoding, all verified against reference implementations.

## What's here

- **GPT-2 (124M) from scratch.** Full architecture, loads OpenAI's pretrained weights, verified against HuggingFace with `torch.allclose` on logits. Also loads GPT-2 Medium (355M) with the same code.
- **Sampling.** Greedy, temperature, top-k, top-p.
- **KV cache.** Verified byte-identical output, benchmarked.
- **Speculative decoding.** 124M drafts for a 355M target, greedy accept/reject, output proven byte-identical to the target's own generation.

## Results

**KV cache (CPU):** per-token time stays flat (~0.025s) as sequence grows. Uncached grows from 0.061s to 0.171s over 20 to 200 tokens. Quadratic to linear, as derived.

**Speculative decoding, acceptance decays with draft depth:**

| k | acceptance rate |
|---|---|
| 2 | 0.79 |
| 4 | 0.69 |
| 8 | 0.54 |

Identical on CPU and GPU, since the math is hardware-independent. Also highly prompt-sensitive: a single trailing space in the prompt shifted acceptance by 10 to 20 points.

**Speculative decoding, speed (s/token, 30 tokens, GPT-2 Medium target):**

| | CPU | GPU (T4) |
|---|---|---|
| spec k=2 | 0.124 | 0.015 |
| spec k=4 | 0.089 | 0.013 |
| spec k=8 | 0.082 | 0.013 |
| plain greedy | 0.073 | 0.012 |

On CPU spec loses everywhere, since there is no parallel-verify discount. On GPU the gap collapses to about 8%. The remaining loss is attributable to two unimplemented optimizations: a target-side KV cache (verify currently re-encodes the full sequence every round) and a persistent drafter cache.
