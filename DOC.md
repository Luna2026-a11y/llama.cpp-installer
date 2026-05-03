🇫🇷 [Français](DOC.fr.md) | 🇬🇧 **English**

# llama.cpp Documentation — Parameters & Internals

> Complete reference for `llama-server` command-line options and the `llama.cpp` engine.

---

## Table of Contents

1. [General Architecture](#general-architecture)
2. [Model Parameters](#model-parameters)
3. [Context Parameters](#context-parameters)
4. [GPU / CUDA Parameters](#gpu--cuda-parameters)
5. [MoE Parameters (Mixture of Experts)](#moe-parameters-mixture-of-experts)
6. [KV Cache Parameters](#kv-cache-parameters)
7. [Sampling / Generation Parameters](#sampling--generation-parameters)
8. [Server Parameters](#server-parameters)
9. [Performance Parameters](#performance-parameters)
10. [Advanced Quantization](#advanced-quantization)
11. [KV Cache Types Deep Dive](#kv-cache-types-deep-dive)
12. [Desktop vs Headless](#desktop-vs-headless)
13. [Internal Mechanics](#internal-mechanics)
14. [Practical Example: Qwen3.6-35B-A3B](#practical-example-qwen36-35b-a3b)
15. [Troubleshooting](#troubleshooting)

---

## General Architecture

### How llama.cpp Works

`llama.cpp` is a C/C++ LLM inference engine optimized for consumer hardware. It does no training — only inference (text generation).

Data flow:

```
GGUF file → Load into memory → Per-layer dequantization → Inference → Generated tokens
```

### Two main executables

| Binary | Role |
|--------|------|
| `llama-server` | HTTP server with OpenAI-compatible API (`/v1/chat/completions`) |
| `llama-cli` | Interactive CLI for terminal chat |

### The GGUF Format

GGUF (GPT-Generated Unified Format) is llama.cpp's model format. It contains:
- Model weights (quantized at Q4, Q5, Q8, etc.)
- Metadata (tokenizer, hyperparameters, chat template)
- Tensors organized by layer

Common quantization levels:

| Quant | Approx. Size | Quality | Recommended Use |
|-------|-------------|---------|-----------------|
| `Q3_K_M` | Smallest | Good | Very limited RAM |
| `IQ4_XS` | ~35% of FP16 | Good | Maximum size savings with imatrix |
| `Q4_K_M` | ~40% of FP16 | Very good | Sweet spot for 12GB GPUs |
| `Q4_K_XL` | Slightly larger | Better | GPUs with more VRAM |
| `Q5_K_M` | ~55% of FP16 | Excellent | When VRAM allows |
| `Q6_K` | ~65% of FP16 | Near-original | Almost lossless |
| `Q8_0` | ~75% of FP16 | Near-perfect | Virtually transparent quantization |
| `UD-Q4_K_XL` | Variable | Unsloth Dynamic | Adaptive quantization (best size/quality ratio) |

> **Note**: `UD-*` (Unsloth Dynamic) quantizations adjust precision per-tensor, offering better quality/size ratios than uniform quantizations. See [Advanced Quantization](#advanced-quantization) for details.

---

## Model Parameters

### `-m, --model <path>` — Path to the GGUF file

Path to the quantized model file.

```bash
-m models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
```

### `--alias <name>` — Model alias

Display name used in the API and logs. Handy when running multiple models.

```bash
--alias qwen35b
```

In the API, the model will be referenced as `qwen35b` instead of the full path.

---

## Context Parameters

### `-c, --ctx-size <n>` — Context size (tokens)

Maximum number of tokens the model can "see" in memory. This is the context window.

```bash
-c 131072   # 128k tokens
-c 32768    # 32k tokens
-c 8192     # 8k tokens (default)
```

**Impact**:
- More context = more VRAM/RAM consumed by the KV cache
- 128k context = ~8 GB of KV cache in q8_0 for a 35B MoE
- Prompt processing speed decreases as context size grows

### `-n, --predict <n>` — Maximum tokens to generate

Upper limit for generation. The model may stop earlier if it encounters an end token.

```bash
-n 32768    # Generate up to 32k tokens
```

### `--no-context-shift` — Disable context shifting

By default, when the context is full, llama.cpp can slide the window (removing the oldest tokens) to continue the conversation. This flag disables that behavior.

**When to use it**: When you want the model to strictly respect the defined context size, without silently truncating history.

---

## GPU / CUDA Parameters

### `-ngl, --n-gpu-layers <n>` — Number of layers on GPU

Defines how many model layers are moved to GPU VRAM. `999` = all layers.

```bash
-ngl 999    # Everything on GPU (recommended if VRAM allows)
-ngl 0      # Everything on CPU (very slow)
-ngl 20     # Only 20 layers on GPU (Partial offloading)
```

**Impact**:
- More GPU layers = faster, but more VRAM needed
- If everything fits in VRAM, use `-ngl 999`
- If insufficient VRAM, reduce progressively until it fits

### `-sm, --split-mode <mode>` — Multi-GPU split mode

| Mode | Description |
|------|-------------|
| `row` | Row-based split (default) — balances between GPUs |
| `layer` | Layer-based split — each GPU handles whole layers |

Only useful with multiple NVIDIA GPUs.

---

## MoE Parameters (Mixture of Experts)

### `-ncmoe, --n-cpu-moe <n>` — Number of MoE layers to keep on CPU

**Key parameter for MoE models on limited VRAM.**

MoE models (like Qwen3.6-35B-A3B, Mixtral, DeepSeek) have multiple "experts" per layer, but only activate a few per token. The challenge: all experts must be in memory to select which ones to activate.

The `-ncmoe` flag controls how many **MoE layers** stay on the **CPU** (offloaded from GPU). Higher values = more MoE layers on CPU = less VRAM used, but slower:

```bash
-ncmoe 0     # All MoE layers on GPU — fastest, but needs more VRAM
-ncmoe 25    # 25 MoE layers on CPU — sweet spot for 12GB VRAM
-ncmoe 999   # All MoE layers on CPU (equivalent to --cpu-moe)
```

> **Note**: The long form is `--n-cpu-moe`, which makes the meaning clear: N layers of MoE weights kept on **CPU**.

**How it works**:

1. The MoE model has N experts per layer (e.g., 64 for Qwen3.6)
2. At each token, only a few experts are activated (e.g., 8 out of 64)
3. But ALL experts must be accessible for selection
4. `-ncmoe 25` keeps the first 25 MoE layers on CPU, the rest on GPU
5. This reduces VRAM usage at the cost of slower expert access for offloaded layers

**Performance impact**:

| Value | VRAM | Speed | Recommendation |
|-------|------|-------|----------------|
| `-ncmoe 0` | ~10.5 GB | 60-65 tok/s | Fits 12GB VRAM comfortably ★ |
| `-ncmoe 25` | ~6.5-10 GB | 50-58 tok/s | Tight 12GB VRAM / safety margin |
| `-ncmoe 999` / `--cpu-moe` | ~4 GB | 25-30 tok/s | Very limited GPU |

> **Tip**: With `--fit on`, llama-server will automatically adjust MoE offloading if VRAM is insufficient. Start with `-ncmoe 0` and let `--fit on` handle it.

---

## KV Cache Parameters

### `-ctk, --cache-type-k <type>` — KV cache quantization type (keys)

### `-ctv, --cache-type-v <type>` — KV cache quantization type (values)

The KV (Key-Value) cache stores intermediate representations of already-processed tokens, so they don't need to be recomputed for each new token.

```bash
-ctk q8_0   # Keys quantized to Q8 (8-bit)
-ctv q8_0   # Values quantized to Q8 (8-bit)
```

**Available types**:

| Type | Precision | VRAM | Long context coherence |
|------|-----------|------|------------------------|
| `f16` | Full | 2x | Reference |
| `q8_0` | 8-bit | 1x | Very good ★ |
| `q4_0` | 4-bit | 0.5x | Notable degradation |
| `q4_1` | Improved 4-bit | 0.5x | Acceptable |

**Why q8_0 is the default sweet spot**:
- Halves KV cache VRAM compared to FP16
- Preserves enough precision for long contexts (128k)
- Degradation is imperceptible in practice
- q4_0 is too aggressive — the model loses coherence beyond 32k tokens

**Impact on 128k context**:
- f16: ~16 GB KV cache → impossible with 12 GB VRAM
- q8_0: ~8 GB KV cache → fits in 12 GB VRAM with margin
- q4_0: ~4 GB but degraded quality

> **See also**: The [KV Cache Types Deep Dive](#kv-cache-types-deep-dive) section for detailed analysis of each cache type, including `turbo3` and desktop-specific considerations.

---

## Sampling / Generation Parameters

### `--temp <value>` — Temperature

Controls model "creativity". Lower is more deterministic.

```bash
--temp 0.6   # Good for coding — creative but coherent
--temp 0.0   # Greedy decoding — always same output
--temp 1.0   # Very random
```

**Recommended values**:
- Coding/analysis: 0.3 - 0.6
- Creative chat: 0.7 - 0.9
- Deterministic tasks: 0.0

### `--top-p <value>` — Nucleus sampling

Only considers tokens whose cumulative probability reaches this threshold.

```bash
--top-p 0.95   # Tokens covering 95% of probability
```

### `--top-k <value>` — Top-K sampling

Only considers the K most probable tokens.

```bash
--top-k 20   # Only the 20 most probable tokens
```

### `--repeat-penalty <value>` — Repetition penalty

Increases penalty for already-generated tokens, preventing loops.

```bash
--repeat-penalty 1.00   # No penalty (recommended for MoE with thinking)
```

> **Note**: For models with "thinking" (like Qwen3), a repetition penalty of 1.0 is recommended because the model naturally manages its thought structure.

### `--presence-penalty <value>` — Presence penalty

Penalizes tokens that have already appeared, encouraging diversity.

```bash
--presence-penalty 0.00   # No additional penalty
```

### `--chat-template-kwargs` — Chat template arguments

Additional parameters passed to the model's chat template.

```bash
--chat-template-kwargs '{"preserve_thinking": true}'
```

`preserve_thinking: true` is **essential** for models with a thinking/reasoning mode (like Qwen3). Without it, thinking content (`<think>...</think>`) is silently stripped from API responses.

---

## Server Parameters

### `--host <address>` — Listen address

```bash
--host 0.0.0.0   # Listen on all interfaces (network access)
--host 127.0.0.1  # Local only
```

### `--port <port>` — Listen port

```bash
--port 8081    # Custom port
--port 8001    # Another port
```

### `--fit` — Auto-adjust resources

```bash
--fit on    # Automatically adjusts parameters if VRAM is insufficient
```

Disables options that cause OOM at startup. Very convenient to avoid manual tuning.

---

## Performance Parameters

### `-fa, --flash-attn` — Flash Attention

```bash
-fa on    # Enable Flash Attention
```

**Flash Attention** is an algorithmic optimization of the attention mechanism that:
- Reduces memory complexity from O(n²) to O(n)
- Accelerates long-context processing
- Is **essential** for contexts > 32k tokens
- Only compatible with NVIDIA GPUs (CUDA)

**Typical gain**: 20-40% faster on long contexts, significant VRAM reduction.

### `-t, --threads <n>` — Number of CPU threads

```bash
-t 8    # 8 CPU threads for non-GPU processing
```

Even with a GPU, non-offloaded parts use the CPU. Optimal number usually matches physical cores.

### `-b, --batch-size <n>` — Batch size for prefill

```bash
-b 512   # Batch of 512 tokens for prompt processing
```

Larger sizes speed up prefill but temporarily consume more VRAM.

---

## Advanced Quantization

Beyond the standard `Q4_K_M` and `Q5_K_M` quantizations, llama.cpp supports advanced quantization methods that offer better size-to-quality tradeoffs — but they come with nuances worth understanding.

### IQ4_XS — Imatrix-Based Quantization

IQ4_XS is an **importance-matrix (imatrix) based** quantization that distributes bits per-tensor rather than uniformly. Instead of every tensor getting the same 4-bit treatment, tensors that matter more for output quality receive higher precision, while less critical tensors are quantized more aggressively.

**Key characteristics**:
- Smaller than `Q4_K_M` (~35% of FP16 vs ~40%) while maintaining competitive quality
- Requires an importance matrix (imatrix) computed during quantization — the imatrix captures which tensors matter most for inference quality
- The per-tensor bit distribution means critical attention and output layers retain more information than uniform Q4

**The cHunter789 QKV Trick**: By default, IQ4_XS quantizations (as produced by the common quantize scripts) force QKV (query/key/value) attention layers to a minimum of `Q5_K` precision regardless of the target quantization level. This preserves attention quality but adds ~0.4 GB to the model size compared to what a strict per-imatrix distribution would produce.

If you want to reclaim that ~0.4 GB (at a small quality cost in attention precision), you can revert the commit that introduced this QKV floor. The trick is commonly referenced as the **cHunter789 QKV override**:

```bash
# When building llama.cpp from source, revert the QKV floor commit
# before compiling, to allow IQ4_XS to quantize QKV layers normally
# rather than forcing Q5_K minimum. Saves ~0.4 GB on 27B-class models.
git revert <QKV-floor-commit-hash>
```

This is an advanced optimization — only consider it if every megabyte of VRAM counts and you can tolerate slightly noisier attention.

### UD (Unsloth Dynamic) Quantizations

UD quantizations (`UD-Q4_K_XL`, `UD-Q4_K_M`, etc.) use a **dynamic per-tensor precision** strategy developed by Unsloth. Rather than picking a single quantization level for the entire model, UD quantizations:

1. **Profile each tensor's importance** using calibration data
2. **Assign higher precision** (e.g., Q6_K, Q5_K) to attention-critical tensors like QKV and output projections
3. **Assign lower precision** (e.g., Q4_K, Q3_K) to less impactful tensors like FFN intermediate layers
4. **Optimize the global size/quality tradeoff** — the result often outperforms a uniform `Q4_K_M` at similar or smaller file sizes

The net effect is that a `UD-Q4_K_XL` model is roughly the same size as `Q4_K_M` but with measurably better perplexity, because bits are allocated where they matter most.

**When to prefer UD**:
- If your GPU can fit `Q5_K_M` but not `Q6_K`, try `UD-Q4_K_XL` — similar quality in less space
- If you're squeezing every token of context, the smaller file size directly frees KV cache VRAM

### Quantization Comparison Table

| Quant | Approx. Size | Quality | Notes |
|-------|-------------|---------|-------|
| `Q3_K_M` | Smallest (~25% FP16) | Good | Last resort for very limited RAM |
| `IQ4_XS` | ~35% of FP16 | Good | Smallest imatrix-based; QKV floor adds ~0.4 GB |
| `Q4_K_M` | ~40% of FP16 | Very good | Standard sweet spot |
| `Q4_K_XL` | Slightly larger | Better | More headroom |
| `Q5_K_M` | ~55% of FP16 | Excellent | High quality, needs more VRAM |
| `Q6_K` | ~65% of FP16 | Near-original | Almost lossless |
| `Q8_0` | ~75% of FP16 | Near-perfect | Virtually transparent |
| `UD-Q4_K_XL` | ~40% of FP16 | Better than Q4_K_M | Best size/quality ratio at this tier |
| `UD-Q4_K_M` | ~38% of FP16 | Similar to Q4_K_M, smaller | Adaptive per-tensor precision |
| `UD-Q5_K_M` | ~55% of FP16 | Better than Q5_K_M | Adaptive per-tensor precision |

---

## KV Cache Types Deep Dive

The KV cache is the single largest variable in your VRAM budget after model weights. Choosing the right cache type determines how much context you can fit and how fast inference runs at depth. This section goes beyond the quick-reference table in [KV Cache Parameters](#kv-cache-parameters) and covers the practical tradeoffs.

### turbo3 KV — Maximum Context

**turbo3** is a hybrid KV cache type that splits precision per-head within each layer: Q/K heads use `q8_0` and V heads use `q4_0`. This gives you significantly more context headroom than pure `q8_0` while keeping key precision higher than pure `q4_0`.

```bash
-ctk q8_0 -ctv turbo3   # Keys at q8_0, values with turbo3 hybrid split
```

Or more commonly both set to turbo3:

```bash
-ctk turbo3 -ctv turbo3
```

**How it works**:
- Each attention head's key is stored at `q8_0` (8-bit)
- Each attention head's value is stored at `q4_0` (4-bit)
- The per-head hybrid approach means keys — which determine attention routing — stay precise
- Values — which are mixed during attention — can tolerate lower precision

**Tradeoff**: turbo3 maximizes context length but **speed degrades at deep context**. Benchmarks on NVIDIA Blackwell GPUs show:

| Context Depth | Speed (turbo3 KV) |
|--------------|-------------------|
| Shallow (≤8k) | ~46 tok/s |
| Mid (≈32k) | ~30 tok/s |
| Deep (≈65k) | ~19 tok/s |

The speed drop is inherent to the lower-precision value cache — at longer sequences, the quantization noise accumulates and the attention mechanism does more work to compensate.

### q4_0 KV — Desktop Sweet Spot

If you're running on a desktop GPU (not headless), `q4_0` KV is often the pragmatic choice:

```bash
-ctk q4_0 -ctv q4_0
```

- **More context** than `q8_0` — roughly twice the context window for the same VRAM
- **Better speed at depth** than `turbo3` — uniform 4-bit is simpler for the kernel to process
- **Acceptable quality** up to ~32k tokens; noticeable degradation beyond that

For desktop users with a 16 GB GPU, `q4_0` KV often provides the best balance of context length and sustained speed in long conversations.

### q8_0 KV — Reference Quality

```bash
-ctk q8_0 -ctv q8_0
```

- **The quality reference**: imperceptible difference from f16 for most tasks
- **Halved VRAM** vs f16 — this is the standard baseline
- **Consistent speed** at all context depths — no degradation curve
- **Limited context** on small GPUs — on a 16 GB card with a 27B model, you'll top out around 55k tokens headless

### f16 KV — Full Precision

```bash
-ctk f16 -ctv f16
```

Full fp16 cache. Highest quality, highest VRAM cost (~2× q8_0). Rarely used outside of benchmarking or quality validation — on most consumer GPUs, the context window is too small to be practical.

### KV Cache Comparison Table (27B Dense Model)

| KV Type | Per-token Cost | Context Limit (16GB headless) | Quality | Speed at Depth |
|---------|---------------|-------------------------------|---------|----------------|
| `f16` | ~8 KB/token | ~27k tokens | Reference | Consistent, fast |
| `q8_0` | ~4 KB/token | ~55k tokens | Near-perfect | Consistent |
| `q4_0` | ~2 KB/token | ~110k tokens | Good to ~32k | Good |
| `turbo3` | ~4 KB/token (K) + ~2 KB/token (V) | ~110k tokens (headless) | Good to ~65k | Degrades at depth |

> **Note**: The per-token cost varies by model architecture (number of heads, head dimension). For a 27B dense model, `q8_0` KV is approximately 4 KB/token. MoE models with shared attention layers may be lower. Always check `llama-server` startup logs for exact memory budgets.

### Desktop vs Headless Context Comparison

The following table shows realistic context limits for a 27B dense model on a **16 GB GPU** (e.g., RTX 4080, RTX 5070 Ti). Desktop figures assume ~500 MiB overhead from the X server and compositor.

| KV Type | Headless Context | Desktop Context |
|---------|-----------------|-----------------|
| `turbo3` / `q4_0` | ~110k tokens | ~75k tokens |
| `q8_0` | ~55k tokens | ~40k tokens |
| `f16` | ~27k tokens | ~20k tokens |

The ~35k token gap between headless and desktop for `turbo3`/`q4_0` comes directly from the ~487 MiB lost to the display server (see [Desktop vs Headless](#desktop-vs-headless) below).

---

## Desktop vs Headless

### The Desktop VRAM Problem

**Headless benchmark numbers do not translate directly to desktop use.** This is one of the most common sources of confusion when configuring llama.cpp.

When you run `nvidia-smi` on a GPU with 16 GB (16304 MiB) of VRAM on an Ubuntu desktop, the actual VRAM available to llama.cpp is significantly less:

```
$ nvidia-smi
# Reports: 16304 MiB total

# But after X server + compositor:
# Actual available: ~15817 MiB
# Overhead:        ~487 MiB
```

That ~500 MiB is consumed by:
- The X.org or Wayland display server
- The GPU compositor (Mutter, KWin, etc.)
- Any desktop rendered on the GPU (even idle desktops consume VRAM for framebuffers)
- GPU-accelerated applications (browser, terminal, etc.)

**The math**: On a 27B dense model with `turbo3` KV, each token of context costs approximately **4 KB**. Losing 487 MiB means:

```
487 MiB × 1024 KiB/MiB ÷ 4 KiB/token ≈ 124,672 tokens
```

In practice the impact is ~35k tokens because the VRAM is split between KV cache and other allocations, and fragmentation occurs. But the principle is clear: **on a desktop, you lose roughly 3% of your context capacity to the display server alone**.

### How to Check Your Available VRAM

**Method 1** — `nvidia-smi` (before starting llama-server):

```bash
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
```

This shows what `nvidia-smi` thinks is free — but includes the display server allocation.

**Method 2** — Check `llama-server` startup logs:

When `llama-server` starts, it prints the VRAM budget ggml sees:

```
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4080, compute capability 8.9, VMM: yes
  ...
  VRAM budget: 15817 MiB
```

The `VRAM budget` line tells you exactly what ggml can allocate — **this is the number that matters**, not the `nvidia-smi` total.

If the ggml VRAM budget is significantly less than `nvidia-smi` reports as total, the difference is your display overhead.

**Method 3** — Run headless to confirm:

```bash
# Stop your display manager temporarily
sudo systemctl stop gdm   # or lightdm, sddm, etc.
# Run llama-server
./llama-server ...
# Check VRAM budget in logs — it should now be close to 16304 MiB
```

### Tips for Desktop Users

- **Use `-ctk q4_0 -ctv q4_0`** or **`-ctk turbo3 -ctv turbo3`** if you need maximum context on a desktop — the reduced per-token cost compensates for the VRAM lost to the display server.
- **Use `--fit on`** to let llama-server automatically reduce context or offload layers if VRAM is insufficient.
- **Close GPU-accelerated applications** (browsers with hardware acceleration, video players, etc.) before starting long-context sessions.
- **Consider running headless** (no display server) for maximum context capacity, accessed via SSH or a separate integrated GPU for display.

---

## Internal Mechanics

### Request Lifecycle

```
1. Client sends POST /v1/chat/completions
2. llama-server tokenizes the prompt
3. Prompt processing (prefill):
   - All prompt tokens are processed in parallel
   - KV cache is populated
   - Flash Attention accelerates this step
4. Token-by-token generation:
   - Model selects active MoE experts for this token
   - KV cache is consulted for context
   - One token is generated
5. Token is returned (streaming or not)
6. Return to step 4 until generation ends
```

### VRAM Layout

On an RTX 4070 12GB with Qwen3.6-35B-A3B (Q4_K_M):

```
Total VRAM : 12 GB
├── Model weights : ~6.5 GB
│   ├── Main layers (GPU) : ~4 GB
│   └── MoE experts (all on GPU with -ncmoe 0) : ~2.5 GB
├── KV cache (q8_0, 128k) : ~4-8 GB (grows with context)
└── CUDA overhead : ~0.5 GB
```

> With `-ncmoe 0`, all MoE layers are on GPU for max speed. If VRAM is tight, use `-ncmoe 25` to offload 25 MoE layers to CPU.

### MoE Experts in Detail

A 35B parameter MoE model with 64 experts only activates a few (8 for Qwen3.6) per token:

```
Token "Hello"  → Selected experts : [3, 17, 22, 31, 45, 51, 58, 61]
Token "world"  → Selected experts : [3, 12, 22, 31, 40, 45, 55, 58]
Token "today"  → Selected experts : [5, 17, 22, 31, 45, 50, 58, 63]
                    ↑                    ↑
                    Frequent experts     Context-specific experts
```

**MoE advantage**: Only activated experts are computed → 35B params but only ~8B computed per token (A3B = Active 3B).

---

## Practical Example: Qwen3.6-35B-A3B

### Recommended Command (RTX 4070 12GB)

```bash
llama-server \
  -m models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --alias qwen35b \
  --host 0.0.0.0 --port 8081 \
  -ngl 999 -ncmoe 0 -fa on \
  -ctk q8_0 -ctv q8_0 \
  -c 131072 -t 8 \
  --no-context-shift \
  --temp 0.6 --top-p 0.95 --top-k 20 \
  --repeat-penalty 1.00 --presence-penalty 0.00 \
  --fit on \
  --chat-template-kwargs '{"preserve_thinking": true}'
```

### Flag-by-flag Explanation

| Flag | Why |
|------|-----|
| `-ngl 999` | All layers on GPU for max performance |
| `-ncmoe 0` | All MoE layers on GPU — fastest (use with `--fit on` for safety) |
| `-fa on` | Flash Attention essential for 128k |
| `-ctk q8_0 -ctv q8_0` | Q8 KV cache — halves VRAM, preserves quality |
| `-c 131072` | Full 128k context |
| `-t 8` | 8 CPU threads for non-GPU parts |
| `--no-context-shift` | No silent context sliding |
| `--temp 0.6` | Moderate creativity, good for coding |
| `--top-p 0.95 --top-k 20` | Conservative sampling |
| `--repeat-penalty 1.00` | No penalty — MoE manages its structure |
| `--presence-penalty 0.00` | No diversity bias |
| `--fit on` | Auto-adjust if VRAM insufficient |
| `--chat-template-kwargs '{"preserve_thinking": true}'` | Preserve thinking blocks |

### Expected Results

| Metric | Value |
|--------|-------|
| Generation speed | 58-62 tok/s |
| VRAM (empty context) | ~6.5 GB |
| VRAM (full 128k context) | ~10.6 GB |
| Prompt processing | ~2000 tok/s |

---

## Troubleshooting

### CUDA OOM (Out of Memory)

```
CUDA error: out of memory
```

Solutions (in order):
1. Increase `-ncmoe` (e.g., `-ncmoe 25` to offload MoE layers to CPU, reducing VRAM)
2. Reduce `-c` (e.g., `-c 65536` instead of `-c 131072`)
3. Switch KV cache to `q4_0` (quality degradation): `-ctk q4_0 -ctv q4_0`
4. Reduce `-ngl` (e.g., `-ngl 80` instead of `-ngl 999`)

### Desktop VRAM Overhead — X Server Eats ~500 MiB

If your model loads headless but OOMs on desktop, the X server and compositor are consuming VRAM that `nvidia-smi` doesn't always surface clearly.

**Symptoms**:
- `nvidia-smi` reports 16304 MiB total, but ggml's VRAM budget is only ~15817 MiB
- The same `-c` setting that works headless causes OOM on desktop
- You lose ~35k tokens of context compared to headless benchmarks

**Solutions**:
- Use `-ctk q4_0 -ctv q4_0` or `-ctk turbo3 -ctv turbo3` to fit more context in less VRAM
- Add `--fit on` to let the server auto-adjust
- Close GPU-accelerated apps before starting llama-server
- Run headless (stop display manager) for maximum capacity

### Slow Inference at Deep Context (turbo3 KV Degradation)

If you're using `turbo3` KV and notice generation speed dropping significantly as context fills up, this is expected behavior.

**Symptoms**:
- Shallow context (≤8k): ~46 tok/s
- Deep context (≈65k): ~19 tok/s

**Solutions**:
- Switch to `-ctk q4_0 -ctv q4_0` for better speed at depth (less context capacity but more consistent speed)
- Switch to `-ctk q8_0 -ctv q8_0` if you can accept fewer tokens — speed stays consistent
- Reduce `-c` to limit the context window and keep speed higher

### Slow Model (General)

- Verify `-fa on` is enabled (Flash Attention is critical for speed)
- Verify `-ngl 999` is used (all layers on GPU)
- Check CPU threads: `-t` should match physical cores
- Verify CUDA is actually used: `nvidia-smi` should show GPU utilization during inference
- Check that your model isn't partially on CPU — look for "offloaded" messages in startup logs

### Model Not Found / Wrong Path

```
error: failed to open model file: models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
```

**Solutions**:
- Use an absolute path: `-m /home/user/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`
- Verify the file exists: `ls -la /path/to/model.gguf`
- Check for typos in the filename (case-sensitive on Linux)
- Ensure the `.gguf` file isn't corrupted: check the file size matches what you downloaded

### Port Already in Use

```
error: failed to bind port 8081: Address already in use
```

**Solutions**:
- Check what's using the port: `ss -tlnp | grep 8081` or `lsof -i :8081`
- Use a different port: `--port 8082`
- Kill the existing process if stale: `kill <PID>`

### nvidia-smi Shows GPU but CUDA Build Fails

```
ggml_cuda_init: CUDA not found
```

Or the `llama-server` binary runs without GPU acceleration despite `nvidia-smi` working.

**Cause**: `nvidia-smi` comes from the NVIDIA driver, but compiling llama.cpp with CUDA support requires the **NVIDIA CUDA Toolkit** (development headers + compiler).

**Solutions**:
- Install the CUDA toolkit: `sudo apt install nvidia-cuda-toolkit` (Ubuntu) or download from [developer.nvidia.com](https://developer.nvidia.com/cuda-downloads)
- Verify `nvcc` is available: `nvcc --version`
- Rebuild llama.cpp: the build script should detect CUDA via `nvcc`
- If building manually, ensure `-DGGML_CUDA=ON` is set in your CMake configuration

### Blackwell GPU — Flash Attention Kernel Note

On NVIDIA Blackwell GPUs (RTX 50-series, B100, B200), the Flash Attention kernel may require a recent build of llama.cpp. Older builds may fall back to slow attention paths or fail silently.

**Solutions**:
- Use the latest release of llama.cpp — Flash Attention support for Blackwell was added recently
- If you see errors referencing `flash_attn` or kernel compilation failures, update llama.cpp
- Verify with `-fa on` and check the startup log for `"flash attention"` confirmation

### Truncated Context

- Verify `-c 131072` in the command
- Verify `--no-context-shift` to prevent silent truncation
- Switch to a lower KV cache type (`q4_0` or `turbo3`) if quality degrades near context limits
- Check your actual VRAM budget in startup logs — you may be running out and ggml is silently shrinking context