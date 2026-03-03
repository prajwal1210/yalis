# API Reference

This document describes the public Python API exposed by the `yalis` package.

---

## ModelConfig

```python
from yalis import ModelConfig
```

Configuration object for model initialization.

### Constructor

```python
ModelConfig(
    model_name: str,
    model_path: Optional[str] = None,
    precision: Literal["fp32", "fp16", "bf16"] = "fp16",
    disable_tp: bool = False,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | required | Hugging Face model ID (e.g. `"meta-llama/Meta-Llama-3-8B-Instruct"`). Used to load the tokenizer. |
| `model_path` | `Optional[str]` | `None` | Path to a converted YALIS checkpoint directory. If `None`, resolved automatically from `$YALIS_CACHE/checkpoints/<model_name>`. |
| `precision` | `str` | `"fp16"` | Floating point precision: `"fp32"`, `"fp16"`, or `"bf16"`. |
| `disable_tp` | `bool` | `False` | Disable tensor parallelism for this model. Useful when running speculative decoding, where the draft and target models may require separate TP configs. |

### Notes

- `model_path` must point to a directory containing a `model_config.yaml` and either a `yalis_checkpoints/` subdirectory (safetensors) or a `lit_model.pth` file (legacy).
- If `model_path` is not provided, the path is constructed as `$YALIS_CACHE/checkpoints/<model_name>`. Ensure the `YALIS_CACHE` environment variable is set correctly.

### Example

```python
model_config = ModelConfig(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    precision="bf16",
)
```

---

## InferenceConfig

```python
from yalis import InferenceConfig
```

Configuration object for inference behavior.

### Constructor

```python
InferenceConfig(
    max_batch_size: int = 1,
    max_length_of_generated_sequences: int = 1024,
    top_k: Optional[int] = None,
    top_p: Optional[float] = 1.0,
    temperature: Optional[float] = 1.0,
    metrics: bool = False,
    tp_dims: Optional[Tuple[int, int, int]] = None,
    attention_backend: str = "flash",
    use_intra_head_parallelism: bool = False,
    use_paged_kv_caching: bool = False,
    prestore_kv_cache: bool = True,
    symmetric_allreduce_strategy: Optional[Literal["one-shot", "two-shot", "nvshmem"]] = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_batch_size` | `int` | `1` | Maximum number of sequences processed simultaneously. The KV cache is pre-allocated for this many sequences. |
| `max_length_of_generated_sequences` | `int` | `1024` | Maximum total sequence length (prompt + generated tokens). |
| `top_k` | `Optional[int]` | `None` | Top-k sampling. `None` disables top-k filtering. |
| `top_p` | `Optional[float]` | `1.0` | Nucleus (top-p) sampling probability. `1.0` disables nucleus filtering. |
| `temperature` | `Optional[float]` | `1.0` | Sampling temperature. `0.0` is greedy decoding. |
| `metrics` | `bool` | `False` | Enable metrics collection (reserved for future use). |
| `tp_dims` | `Optional[Tuple[int, int, int]]` | `None` | 3-dimensional tensor parallelism layout. See [Tensor Parallelism](tensor_parallelism.md). |
| `attention_backend` | `str` | `"flash"` | Attention implementation: `"flash"`, `"sdpa"`, or `"flex"`. |
| `use_intra_head_parallelism` | `bool` | `False` | Enable intra-head attention parallelism. Requires `attention_backend="sdpa"`. |
| `use_paged_kv_caching` | `bool` | `False` | Use paged KV cache for memory-efficient attention. Requires `attention_backend="flash"`. |
| `prestore_kv_cache` | `bool` | `True` | Pre-store KV cache entries before the attention computation. |
| `symmetric_allreduce_strategy` | `Optional[str]` | `None` | All-reduce strategy for tensor parallelism: `"one-shot"`, `"two-shot"`, or `"nvshmem"`. `None` uses the default NCCL all-reduce. |

### Constraints

- `use_paged_kv_caching=True` requires `attention_backend="flash"`.
- `use_intra_head_parallelism=True` requires `attention_backend="sdpa"`.
- `torch >= 2.6.0` is required.

### Example

```python
inference_config = InferenceConfig(
    max_batch_size=16,
    max_length_of_generated_sequences=2048,
    temperature=0.8,
    top_p=0.9,
    attention_backend="flash",
    use_paged_kv_caching=False,
)
```

---

## LLMEngine

```python
from yalis import LLMEngine
```

The main inference engine. Loads a model, manages the KV cache, and handles batched generation.

### Constructor

```python
LLMEngine(
    model_config: ModelConfig,
    inference_config: InferenceConfig,
    device: str = "cuda",
)
```

Initializes the distributed backend, loads the model from disk, allocates KV cache, and sets up the tokenizer.

### Methods

#### `generate`

```python
engine.generate(
    prompts: Union[list[str], list[list[int]]],
    tokens_to_generate: int = 50,
    report_throughput: bool = False,
    ignore_eos: Optional[bool] = True,
    enable_nvtx: Optional[bool] = False,
    get_logits: Optional[bool] = False,
) -> Tuple[torch.Tensor, dict] | Tuple[torch.Tensor, dict, list]
```

Generate tokens for a batch of prompts.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompts` | `list[str]` or `list[list[int]]` | required | List of string prompts (auto-tokenized) or pre-tokenized integer ID lists. |
| `tokens_to_generate` | `int` | `50` | Number of new tokens to generate per prompt. Silently capped at `max_length - prompt_length`. |
| `report_throughput` | `bool` | `False` | Print throughput metrics to stdout on rank 0. |
| `ignore_eos` | `bool` | `True` | If `False`, stop generation when EOS is produced and pad shorter sequences. |
| `enable_nvtx` | `bool` | `False` | Annotate prefill/decode steps with NVTX ranges for profiling. |
| `get_logits` | `bool` | `False` | Also return per-step logits. |

**Returns:**
- `(output_tensor, metrics)` — `output_tensor` has shape `(batch_size, tokens_to_generate)`.
- `(output_tensor, metrics, output_logits)` — when `get_logits=True`.

**`metrics` dict keys:**

| Key | Description |
|-----|-------------|
| `BatchSize` | Number of sequences in the batch |
| `PromptLength` | Padded prompt length |
| `DecodeLength` | Number of tokens generated |
| `Throughput` | Tokens per second |
| `TTFT` | Time to first token (ms) |
| `TBT` | Time between tokens, averaged over decode steps (ms) |
| `E2E` | End-to-end generation time (ms) |
| `TokenizationTime` | Time spent tokenizing (ms) |
| `FinishedReason` | `"EOS"` or `"Max Token Length"` |

#### `reset_kv_cache`

```python
engine.reset_kv_cache(max_batch_size: int)
```

Clear and re-allocate the KV cache for a different `max_batch_size`. Useful when switching between batch sizes at runtime.

### Example

```python
from yalis import ModelConfig, InferenceConfig, LLMEngine

model_config = ModelConfig(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    precision="bf16",
)
inference_config = InferenceConfig(
    max_batch_size=8,
    max_length_of_generated_sequences=1024,
    temperature=1.0,
    top_p=0.8,
    attention_backend="flash",
)

engine = LLMEngine(model_config=model_config, inference_config=inference_config)

output_tokens, metrics = engine.generate(
    prompts=["What is the capital of France?"],
    tokens_to_generate=128,
    report_throughput=True,
)
```

---

## SpeculativeLLMEngine

```python
from yalis import SpeculativeLLMEngine
```

Extends `LLMEngine` with draft-based speculative decoding. Loads both a target and a draft model, and uses rejection sampling to verify draft tokens.

See [Speculative Decoding](speculative_decoding.md) for a full guide.

### Constructor

```python
SpeculativeLLMEngine(
    target_model_config: ModelConfig,
    draft_model_config: ModelConfig,
    inference_config: InferenceConfig,
    device: str = "cuda",
)
```

### Constraints

- `use_paged_kv_caching` is **not** supported.
- `attention_backend` must be `"flash"` or `"sdpa"`.

### Methods

#### `generate_speculative`

```python
engine.generate_speculative(
    input_tokens: Union[list[str], list[list[int]]],
    tokens_to_generate: int,
    gamma: int,
    report_throughput: bool = False,
    ignore_eos: bool = True,
    enable_nvtx: bool = False,
) -> Tuple[torch.Tensor, dict]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_tokens` | `list[str]` or `list[list[int]]` | Input prompts. |
| `tokens_to_generate` | `int` | Number of tokens to generate. |
| `gamma` | `int` | Number of draft tokens proposed per speculative step. |
| `report_throughput` | `bool` | Print throughput metrics on rank 0. |
| `ignore_eos` | `bool` | If `False`, stop at EOS. |
| `enable_nvtx` | `bool` | Enable NVTX annotations for profiling. |

**`metrics` dict keys** (in addition to the standard ones):

| Key | Description |
|-----|-------------|
| `TBS` | Time per speculative step (ms) |
| `TBS (Draft)` | Time spent in draft decode per step (ms) |
| `TBS (Verify)` | Time spent in target verify per step (ms) |
| `AcceptanceRate` | Fraction of draft tokens accepted by the target model |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `YALIS_CACHE` | Root directory for model checkpoints. Checkpoints are stored under `$YALIS_CACHE/checkpoints/<model_name>/`. MoE kernel configs are stored under `$YALIS_CACHE/configs/`. |
| `YALIS_DISABLE_COMPILE` | Set to `"1"` to disable `torch.compile` (useful for debugging). |
| `YALIS_DISABLE_DECODE_CUDAGRAPHS` | Set to `"1"` to disable CUDA graph capture during decode (uses `mode="default"` instead of `mode="reduce-overhead"`). |
| `HF_HOME` | Hugging Face cache directory (for tokenizers and HF model downloads). |
| `HF_TOKEN` | Hugging Face API token. Required for gated models (e.g. Meta LLaMA). |
