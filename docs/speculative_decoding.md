# Speculative Decoding

Speculative decoding accelerates LLM inference by using a small, fast **draft model** to propose multiple tokens at once, which a larger **target model** then verifies in a single forward pass. When the draft tokens are accepted, the engine produces `gamma` tokens for the cost of roughly one target-model forward pass.

YALIS implements draft-based speculative decoding in `SpeculativeLLMEngine`.

---

## How It Works

Each decode step proceeds as follows:

1. **Draft**: The draft model autoregressively proposes `gamma` candidate tokens.
2. **Verify**: The target model scores all `gamma` draft tokens in a single batched forward pass.
3. **Reject/Accept**: A rejection sampler compares the draft and target probability distributions. Tokens that the target model would have sampled with high probability are accepted; the rest are rejected.
4. **Rewind**: The KV cache of both models is rewound by the number of rejected tokens, so the next step starts from the correct position.
5. **Bonus token**: If all `gamma` draft tokens are accepted, one additional target-model token is generated as a "bonus".

The acceptance rate (fraction of draft tokens accepted) determines the speedup. Higher acceptance rates mean more tokens per target-model call.

---

## Requirements

- `attention_backend` must be `"flash"` or `"sdpa"`.
- `use_paged_kv_caching=False` (paged KV cache is not supported for speculative decoding).
- Both the draft and target model checkpoints must be downloaded and converted.
- The draft model must have a **smaller** tensor parallelism degree than the target model, or use `disable_tp=True` on the draft config if you want it to run without TP.

---

## Usage

```python
from yalis import ModelConfig, InferenceConfig, SpeculativeLLMEngine

# Target: large model
target_config = ModelConfig(
    model_name="meta-llama/Llama-3.1-70B-Instruct",
    precision="bf16",
)

# Draft: small model of the same family
draft_config = ModelConfig(
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    precision="bf16",
    disable_tp=True,  # run draft model without TP
)

inference_config = InferenceConfig(
    max_batch_size=4,
    max_length_of_generated_sequences=1024,
    temperature=1.0,
    top_p=0.9,
    attention_backend="flash",
    use_paged_kv_caching=False,
)

engine = SpeculativeLLMEngine(
    target_model_config=target_config,
    draft_model_config=draft_config,
    inference_config=inference_config,
)

output_tokens, metrics = engine.generate_speculative(
    input_tokens=["Explain the theory of relativity in simple terms."],
    tokens_to_generate=256,
    gamma=4,                # propose 4 draft tokens per step
    report_throughput=True,
)

print(f"Acceptance rate: {metrics['AcceptanceRate']:.2%}")
print(f"Throughput: {metrics['Throughput']:.1f} tok/s")
```

---

## Choosing `gamma`

`gamma` is the number of draft tokens proposed per step. The right value depends on the draft model's acceptance rate for your workload:

| Acceptance rate | Recommended `gamma` |
|----------------|---------------------|
| > 80% | 6–8 |
| 60–80% | 4–6 |
| < 60% | 2–4 |

A `gamma` that is too large wastes compute when many tokens are rejected. Start with `gamma=4` and tune based on the reported `AcceptanceRate`.

---

## Metrics

`generate_speculative` returns a `metrics` dict with these additional keys beyond the standard ones:

| Key | Description |
|-----|-------------|
| `AcceptanceRate` | Fraction of draft tokens accepted (0.0–1.0). Values above 0.7 indicate a well-matched draft model. |
| `TBS` | Average time per speculative decode step (ms). |
| `TBS (Draft)` | Time spent in draft model decode per step (ms). |
| `TBS (Verify)` | Time spent in target model verify per step (ms). |

---

## Tips

- **Model family matching**: Use a draft model from the same family as the target (e.g. Llama 3.2 1B as a draft for Llama 3.1 8B or 70B). Mismatched tokenizers will cause errors.
- **Batch size**: Speculative decoding is most beneficial at batch size 1 or small batches where the target model is memory-bandwidth bound. At large batch sizes, the verify step becomes compute-bound and speedup diminishes.
- **Temperature = 0**: When using greedy decoding (`temperature=0`), acceptance rates tend to be higher because the draft and target models are more likely to agree on the top token.
