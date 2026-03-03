# Tensor Parallelism

YALIS supports multi-GPU inference through tensor parallelism (TP), powered by the [AxoNN](https://github.com/axonn-ai/axonn) library. This document explains how to configure and use TP for large-scale inference.

---

## Overview

Tensor parallelism shards model weight matrices across GPUs so that each GPU holds and computes only a slice of each layer. All-reduce operations synchronize activations between GPUs after each sharded computation. This allows models that are too large to fit on a single GPU to be run across multiple GPUs.

YALIS uses TP automatically when `world_size > 1`. You can disable it per-model with `disable_tp=True` in `ModelConfig`.

---

## Single-Node Multi-GPU

To run on 4 GPUs on one node, launch your script via `torchrun` or SLURM's `srun`:

```bash
# torchrun (local, single-node)
torchrun --nproc_per_node=4 examples/infer.py
```

No code changes are needed — YALIS detects `world_size` automatically and enables TP.

### InferenceConfig for single-node TP

```python
inference_config = InferenceConfig(
    max_batch_size=8,
    max_length_of_generated_sequences=2048,
    attention_backend="flash",
    tp_dims=None,   # default: all GPUs in the first dimension → (4, 1, 1)
)
```

Setting `tp_dims=None` maps all available GPUs into a 1D tensor parallel group: `(world_size, 1, 1)`.

---

## tp_dims: The Process Mesh

`tp_dims` is a 3-tuple `(G_intra_r, G_intra_c, G_intra_d)` that defines the AxoNN process mesh. The product `G_intra_r × G_intra_c × G_intra_d` must equal `world_size`.

| Dimension | Controls |
|-----------|----------|
| `G_intra_r` | Row parallelism (standard column/row linear sharding) |
| `G_intra_c` | Column parallelism |
| `G_intra_d` | Depth parallelism (currently reserved) |

For most use cases, set `tp_dims=(world_size, 1, 1)` or leave it as `None`.

### Example: 8 GPUs in a 1D layout

```python
inference_config = InferenceConfig(
    tp_dims=(8, 1, 1),
    ...
)
```

---

## Multi-Node Inference

For multi-node runs (e.g. on Perlmutter or Polaris), use the provided SLURM scripts as a reference:

```bash
# On Perlmutter — request 2 nodes (8 GPUs total)
salloc --nodes 2 --qos interactive --time 01:00:00 --constraint gpu --gpus 8 --account=m4641_g

# Then submit or run:
bash scripts/run_pm.sh
```

The script sets up the required environment variables and launches the job via `srun`:

```bash
srun -C gpu -N $NNODES -n $GPUS -c 32 --cpu-bind=cores --gpus-per-node=4 \
    ./scripts/get_rank.sh python -u examples/infer.py
```

`get_rank.sh` sets `RANK` and `LOCAL_RANK` correctly for each process. `MASTER_ADDR` and `MASTER_PORT` must be set in the environment before the job starts (handled in `run_pm.sh`).

---

## Intra-Head Parallelism

In addition to standard column/row parallelism, YALIS supports **intra-head attention parallelism**, which further shards individual attention heads across GPUs. This is useful for models with a small number of heads (e.g. GQA models).

To enable:

```python
inference_config = InferenceConfig(
    attention_backend="sdpa",        # required
    use_intra_head_parallelism=True,
    ...
)
```

> Note: `use_intra_head_parallelism` is incompatible with `attention_backend="flash"`.

---

## Symmetric All-Reduce Strategies

For single-node inference, YALIS supports faster all-reduce strategies that exploit NVLink or NVSHMEM shared memory instead of going through the NCCL default path:

```python
inference_config = InferenceConfig(
    symmetric_allreduce_strategy="one-shot",  # or "two-shot" or "nvshmem"
    ...
)
```

| Strategy | Description |
|----------|-------------|
| `None` | Default NCCL all-reduce (works everywhere) |
| `"one-shot"` | Single-round symmetric all-reduce over NVLink. Best for small tensors. |
| `"two-shot"` | Two-round symmetric all-reduce. Better for larger tensors. |
| `"nvshmem"` | NVSHMEM-based all-reduce. Requires NVSHMEM to be installed and available. |

These strategies are only effective on single-node setups with fast GPU interconnects (NVLink).

---

## MoE Models

For Mixture-of-Experts models (e.g. Qwen3-MoE), YALIS shards expert weights across GPUs using `yalis/tensor_parallel/moe.py`. The fused MoE kernel (based on vLLM's Triton implementation) handles expert routing efficiently.

Before running a MoE model for the first time, tune the kernel configuration for your hardware:

```bash
python benchmarks/benchmark_moe.py \
    --num-experts 64 \
    --hidden-size 4096 \
    --intermediate-size 2048 \
    --dtype bf16
```

This writes optimal Triton kernel configs to `$YALIS_CACHE/configs/`, which are loaded automatically at runtime.

---

## Troubleshooting

**OOM during NCCL init**: If you see out-of-memory errors during distributed initialization, it may be due to NCCL allocating memory on the default device. YALIS deliberately avoids passing `device_id` to `init_process_group` to work around this. Ensure you have enough free GPU memory before launching.

**Incorrect rank mapping**: On multi-node SLURM jobs, use the provided `scripts/get_rank.sh` (or `scripts/get_rank_polaris.sh` for Polaris) to ensure `RANK`, `LOCAL_RANK`, and `WORLD_SIZE` are set correctly for each process.
