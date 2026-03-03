# Architecture Overview

This document describes the high-level architecture of YALIS and how its major components interact.

---

## Component Map

```
yalis/
├── engine.py          # LLMEngine, SpeculativeLLMEngine — public entry points
├── config.py          # ModelConfig, InferenceConfig
├── model.py           # get_model() — loads GPT model with TP and attention settings
├── model_loading.py   # Safetensors checkpoint loader
├── initialize.py      # Distributed process group initialization
├── timers.py          # Hierarchical CUDA event timers
├── utils.py           # print_rank0, GPU memory helpers
├── constants.py       # EnginePhase enum
├── attention/         # Attention backends
│   ├── backends.py        # AttentionBackend enum
│   ├── registry.py        # @register_attention decorator
│   ├── flash.py           # FlashAttention-2 backend
│   ├── sdpa_and_flex.py   # SDPA and Flex backends
│   ├── masking.py         # Causal mask helpers
│   ├── update_kv_cache.py # KV cache update ops
│   └── paged_kv_cache.cpp # C++ paged KV cache manager
├── tensor_parallel/   # Tensor parallelism primitives
│   ├── linear.py          # Column/row parallel linear layers
│   ├── all_reduce_op.py   # All-reduce operations
│   ├── rms_norm.py        # RMSNorm with TP support
│   ├── moe.py             # MoE expert parallelism
│   └── nvshmem_comm.py    # NVSHMEM-based symmetric all-reduce
└── external/          # Third-party / adapted code
    ├── model.py           # GPT model definition (adapted from LitGPT)
    ├── config.py          # Model architecture config (from LitGPT)
    ├── litgpt_utils.py    # Checkpoint loading utilities
    ├── sampling.py        # top-k / top-p / greedy sampling
    ├── rejection_sampler.py # Speculative decoding rejection sampler
    ├── fused_moe.py       # Fused MoE Triton kernel (from vLLM)
    ├── nccl_comm.py       # Custom NCCL communicators
    ├── safetensor_saver.py # Checkpoint conversion utilities
    ├── download.py        # HuggingFace model downloader + converter
    └── csrc/vllm/         # CUDA extensions (from vLLM)
```

---

## Inference Lifecycle

### Standard Generation

```
LLMEngine.generate()
    │
    ├─ _tokenize_prompts()      # tokenize strings → (tokens, seq_lengths)
    ├─ _validate_sequence_lengths()
    │
    ├─ prefill()                # step 0: process full prompt → first token
    │   └─ model(tokens, PREFILL, seq_lengths)
    │       └─ GPT.forward() with causal masking over prompt positions
    │
    └─ generate() × N          # steps 1..N: decode one token at a time
        └─ model(tokens, DECODE_SINGLE)
            └─ GPT.forward() using cached KV values
```

Both `prefill` and `generate` are decorated with `@torch.compile`, allowing the decode loop to be captured as CUDA graphs (`mode="reduce-overhead"`) for low-latency inference.

### Speculative Generation

```
SpeculativeLLMEngine.generate_speculative()
    │
    ├─ prefill(draft_model, ...)  # warm up draft KV cache
    ├─ prefill(target_model, ...) # warm up target KV cache → first token
    │
    └─ while not done:
        ├─ generate(draft_model) × gamma   # propose gamma draft tokens
        ├─ verify(target_model, draft_tokens)  # score all drafts in one pass
        ├─ RejectionSampler()              # accept/reject each draft token
        └─ model.rewind_kv_cache(rejected) # roll back KV cache for rejections
```

The `verify` function runs the target model over the full draft sequence in a single forward pass, making speculative decoding efficient at low batch sizes.

---

## Attention Backends

YALIS uses a registry pattern to select the attention implementation at model-load time. The `InferenceConfig.attention_backend` field controls which backend is used.

| Backend | Description | Constraints |
|---------|-------------|-------------|
| `flash` | FlashAttention-2 via the `flash-attn` package | Supports paged KV caching |
| `sdpa` | PyTorch scaled dot-product attention (`F.scaled_dot_product_attention`) | Supports intra-head parallelism |
| `flex` | PyTorch FlexAttention (experimental) | No additional constraints |

The backend is wired in at model construction inside `get_model()` and propagated through the model config.

---

## Tensor Parallelism

YALIS supports two forms of tensor parallelism:

1. **1D column/row parallelism** — shards attention and MLP weight matrices across the GPU world. Activated automatically when `world_size > 1` and `disable_tp=False`.

2. **2D (intra-head) parallelism** — additionally shards attention heads across a second GPU dimension. Enabled via `InferenceConfig.use_intra_head_parallelism=True` with `attention_backend="sdpa"`.

The TP layout can be explicitly controlled via `InferenceConfig.tp_dims`, which takes a 3-tuple `(d0, d1, d2)` describing the process mesh. See [Tensor Parallelism](tensor_parallelism.md) for details.

For **MoE models** (e.g. Qwen3-MoE), `yalis/tensor_parallel/moe.py` handles expert routing and sharding across GPUs.

---

## KV Cache

YALIS supports two KV cache modes:

- **Standard (contiguous)** — a fixed-size `(batch_size, seq_length, n_heads, head_dim)` tensor allocated at startup. Simple and fast.
- **Paged** — memory is managed in fixed-size pages by a C++ extension (`kvcache_manager`). Reduces memory fragmentation at large batch sizes. Only available with `attention_backend="flash"`.

The `model.set_kv_cache()` / `model.clear_kv_cache()` / `model.rewind_kv_cache()` interface is used by the engine to manage cache state across generation steps.

---

## Checkpoint Format

YALIS uses checkpoints converted from Hugging Face format via `yalis/external/download.py`. The conversion produces:

```
<model_name>/
├── model_config.yaml       # architecture hyperparameters
└── yalis_checkpoints/      # sharded safetensors weights
    ├── model-00001-of-NNNNN.safetensors
    └── ...
```

Use `python yalis/external/download.py <hf_model_id>` to download and convert a model. See the [README](../README.md) for the full workflow.
