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
10. [Internal Mechanics](#internal-mechanics)
11. [Practical Example: Qwen3.6-35B-A3B](#practical-example-qwen36-35b-a3b)
12. [Troubleshooting](#troubleshooting)

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
| `Q4_K_M` | ~40% of FP16 | Very good | Sweet spot for 12GB GPUs |
| `Q4_K_XL` | Slightly larger | Better | GPUs with more VRAM |
| `Q5_K_M` | ~55% of FP16 | Excellent | When VRAM allows |
| `Q6_K` | ~65% of FP16 | Near-original | Almost lossless |
| `Q8_0` | ~75% of FP16 | Near-perfect | Virtually transparent quantization |
| `UD-Q4_K_XL` | Variable | Unsloth Dynamic | Adaptive quantization (best size/quality ratio) |

> **Note**: `UD-*` (Unsloth Dynamic) quantizations adjust precision per-tensor, offering better quality/size ratios than uniform quantizations.

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
- 128k context = ~8 GB of KV cache in Q8_0 for a 35B MoE
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

### `--cache-type-k <type>` — KV cache quantization type (keys)

### `--cache-type-v <type>` — KV cache quantization type (values)

The KV (Key-Value) cache stores intermediate representations of already-processed tokens, so they don't need to be recomputed for each new token.

```bash
--cache-type-k q8_0   # Keys quantized to Q8 (8-bit)
--cache-type-v q8_0   # Values quantized to Q8 (8-bit)
```

**Available types**:

| Type | Precision | VRAM | Long context coherence |
|------|-----------|------|------------------------|
| `f16` | Full | 2x | Reference |
| `q8_0` | 8-bit | 1x | Very good ★ |
| `q4_0` | 4-bit | 0.5x | Notable degradation |
| `q4_1` | Improved 4-bit | 0.5x | Acceptable |

**Why Q8_0 is the sweet spot**:
- Halves KV cache VRAM compared to FP16
- Preserves enough precision for long contexts (128k)
- Degradation is imperceptible in practice
- Q4_0 is too aggressive — the model loses coherence beyond 32k tokens

**Impact on 128k context**:
- FP16: ~16 GB KV cache → impossible with 12 GB VRAM
- Q8_0: ~8 GB KV cache → fits in 12 GB VRAM with margin
- Q4_0: ~4 GB but degraded quality

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
--host 127.0.0.1 # Local only
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
├── KV cache (Q8_0, 128k) : ~4-8 GB (grows with context)
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
  --cache-type-k q8_0 --cache-type-v q8_0 \
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
| `--cache-type-k/v q8_0` | Q8 KV cache — halves VRAM, preserves quality |
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
2. Reduce `-c` (e.g., 65536 instead of 131072)
3. Switch KV cache to `q4_0` (quality degradation)
4. Reduce `-ngl` (e.g., 80 instead of 999)

### Slow Model

- Verify `-fa on` is enabled
- Verify `-ngl 999` is used
- Check CPU threads: `-t` should match physical cores
- Verify CUDA is actually used: `nvidia-smi` should show GPU utilization

### Truncated Context

- Verify `-c 131072` in the command
- Verify `--no-context-shift` to prevent silent truncation
- Increase `--cache-type` if quality degrades at end of context