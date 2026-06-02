# MALTOS

**Modular Assembly LLM Training and Optimization Systems.**

MALTOS is a modular, composable runtime for large-scale foundation model
training. Tensor parallelism, sequence parallelism, DDP, ZeRO, precision,
checkpointing, metrics, and profiler traces are assembled as plugins around one
phase-oriented training runtime instead of being baked into the trainer loop.

The goal of this repo is not to hide PyTorch behind a framework. The goal is to
make the moving pieces of a training system explicit: process meshes, runtime
phases, composable parallel plugins, sharded checkpointing, dataloader state,
metric aggregation, and a trainer loop that can run real token shards.

Technical writeups live in the companion blog repo:
[xing7code/maltos-blog](https://github.com/xing7code/maltos-blog).

Experiment tracking: [W&B report](https://api.wandb.ai/links/xing7-org/f2s88x30)

## What Works

- Runtime plugin system with dependency ordering and phase hooks.
- Data parallel plugins: sync DDP, async DDP, and bucketed DDP.
- Tensor parallel and sequence parallel layers for the tiny transformer path.
- ZeRO-1, ZeRO-2, and ZeRO-3 style optimizer/parameter sharding.
- Mixed precision hooks for bf16/fp16, with GradScaler state checkpointing for fp16.
- Gradient accumulation and gradient clipping.
- PyTorch fused AdamW support for CUDA optimizer-step throughput.
- Stateful pretraining dataloader over mmap token shards.
- Sharded checkpoint save/load for model, optimizer, trainer, plugin, RNG, and dataloader state.
- Metric collection from runtime/plugins, interval aggregation, and console/jsonl logging.
- Console, JSONL, and W&B metric logging.
- End-to-end LLaMA/tiny pretraining recipe.

This is intentionally small enough to read, but the core control flow mirrors
the shape of larger pretraining systems: Megatron-style TP/SP, ZeRO/FSDP-style
optimizer ownership, explicit process mesh axes, and checkpoint metadata that
describes local shards. Long term, MALTOS is meant to grow from pretraining into
a modular training stack for SFT, preference training, RL, and fast research
workflows.

## Support Matrix

| Area | Status |
|---|---|
| Single-process training | Supported |
| Sync / async / bucketed DDP | Supported |
| Tensor parallelism | Supported |
| Sequence parallelism | Supported |
| ZeRO-1 / ZeRO-2 / ZeRO-3 style sharding | Supported |
| BF16 / FP16 precision hooks | Supported |
| Gradient accumulation / clipping | Supported |
| Stateful token-shard dataloader | Supported |
| Sharded checkpoint save/load | Supported |
| W&B metric logging and checkpoint artifacts | Supported |
| Pipeline parallelism | Planned |
| Context parallelism | Planned |
| Expert parallelism | Planned |
| LLaMA activation checkpointing | Supported |
| LLaMA SDPA attention backends | Supported |
| FlashAttention-specific kernels | Planned |

## MALTOS Runtime Flow

```mermaid
flowchart LR
    shards["Token .bin shards"] --> loader["PretrainingDataLoader"]
    loader --> trainer["Trainer"]

    trainer --> runtime["MALTOS RuntimeCore"]
    runtime --> model["PyTorch model"]
    runtime --> plugins["Runtime plugins"]
    runtime --> state["StateManager"]

    plugins --> tp["TP / SP"]
    plugins --> dp["DDP / bucket DDP"]
    plugins --> zero["ZeRO-1 / ZeRO-2 / ZeRO-3"]
    plugins --> precision["bf16 / fp16"]
    plugins --> clip["grad clip"]

    trainer --> metrics["MetricAggregator"]
    runtime --> metrics
    plugins --> metrics
    metrics --> logger["Console / JSONL logger"]

    trainer --> ckpt["Checkpoint IO"]
    state --> ckpt
    loader --> state
```

## Runtime Step

`Trainer` owns optimizer-step cadence. `RuntimeCore.run_step()` executes one
logical training microstep: forward, backward, plugin phases, and gradient
accumulation scaling. It returns `(loss, should_step)` so the trainer can
decide whether to call `RuntimeCore.step_optimizer()` at the accumulation
boundary. `StepContext` tracks only the execution cursor needed by the current
runtime:

```python
loss, should_step = runtime.run_step(batch)
if should_step:
    runtime.step_optimizer()
```

`StepContext` currently carries:

- `step`
- `microbatch_idx`
- `grad_accum_steps`

Future parallel strategies such as PP can override `build_step_runner()`, but
the default runtime path is still a single forward/backward implementation:

```mermaid
sequenceDiagram
    participant T as Trainer
    participant R as RuntimeCore
    participant P as Plugins
    participant M as Model
    participant O as Optimizer

    T->>R: run_step(batch)
    R->>P: PRE_MICROBATCH
    R->>P: PRE_FORWARD
    R->>M: forward(batch)
    R->>P: POST_FORWARD
    R->>P: PRE_BACKWARD
    R->>M: backward(loss / grad_accum_steps)
    R->>P: POST_BACKWARD
    alt accumulation boundary
        T->>R: step_optimizer()
        R->>P: PRE_STEP
        R->>O: optimizer.step()
        R->>P: POST_STEP
        R->>R: step += 1
    end
    T->>R: collect_metrics()
```

The trainer collects metrics every microstep, but only logs/checkpoints on
optimizer-step boundaries. This keeps gradient accumulation observability
honest without making checkpoints land mid-step unless explicitly requested by
tests.

## Batch Contract

The pretraining path passes dataloader batches directly through the trainer/runtime into the model:

```python
{
    "input_ids": Tensor[batch, seq],
    "labels": Tensor[batch, seq],
}
```

`TinyTransformer.forward()` also accepts `(input_ids, labels)` for tests and lower-level runtime checks. In both cases, labels are already aligned with logits; the model does not apply an extra causal shift. Test-only helpers that synthesize shifted labels live under `tests/`, not in the model package.

## Repository Layout

```text
data/       Stateful tensor and token-shard dataloaders
models/     TinyModel, TinyTransformer, and LLaMA variants with TP/SP specs
parallel/   ParallelPlan and TP/SP sharding specs
runtime/    RuntimeCore, MeshConfig/group management, plugin API, layers, plugins
state/      StateManager and sharded checkpoint IO
train/      Trainer loop
utils/      Logging and metric aggregation utilities
tools/      Dataset prep and pretraining entrypoints
tests/      Equivalence, checkpoint, integration, and resume tests
docs/       Architecture notes
```

## Quick Start

Install dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Run a tiny single-process pretraining smoke using committed token shards:

```bash
PYTHONPATH=. .venv/bin/python tools/pretrain.py \
  --model tiny \
  --data tests/testdata \
  --vocab-size 256 \
  --dim 32 \
  --n-heads 4 \
  --hidden-size 64 \
  --n-layers 1 \
  --seq-len 16 \
  --micro-batch-size 1 \
  --max-steps 2 \
  --log-every 1
```

Run the core smoke tests:

```bash
PYTHONPATH=. .venv/bin/python tests/smoke_runtime_core.py
PYTHONPATH=. .venv/bin/python tests/smoke_trainer_loop.py
```

Run a TP equivalence test:

```bash
PYTHONPATH=. .venv/bin/python tests/tiny_transformer_tp_runtime_core_equivalence.py \
  --world-size 2 \
  --tp-size 2
```

Run a heavier integration case:

```bash
PYTHONPATH=. .venv/bin/python tests/pretraining_loader_tp_sp_zero3_bf16_clip_accum2_resume.py \
  --world-size 4 \
  --dp-size 2 \
  --tp-size 2
```

That test exercises:

```text
PretrainingDataLoader + TP + SP + ZeRO-3 + bf16 + grad clip
+ gradient accumulation + checkpoint save/load + dataloader resume
```

## Preparing Token Shards

The runtime dataloader consumes raw `.bin` token shards. To tokenize a Hugging Face dataset:

```bash
PYTHONPATH=. .venv/bin/python tools/prepare_token_shards.py \
  --dataset HuggingFaceFW/fineweb-edu \
  --config sample-10BT \
  --split train \
  --column text \
  --tokenizer-name-or-path NousResearch/Llama-2-7b-hf \
  --expected-vocab-size 32000 \
  --output-dir datasets/fineweb_500m \
  --max-tokens 500000000 \
  --tokens-per-shard 100000000 \
  --streaming
```

Or use:

```bash
bash tools/data.sh
```

## Tiny Pretraining Recipe

Single-process LLaMA smoke:

```bash
PYTHONPATH=. .venv/bin/python tools/pretrain.py \
  --model llama \
  --data tests/testdata \
  --vocab-size 256 \
  --dim 32 \
  --n-heads 4 \
  --hidden-size 64 \
  --n-layers 1 \
  --seq-len 16 \
  --micro-batch-size 1 \
  --max-steps 20 \
  --metrics-jsonl logs/llama_smoke.jsonl
```

YAML recipes are supported for real runs:

```bash
PYTHONPATH=. .venv/bin/python tools/pretrain.py \
  --config configs/llama_10m.yaml \
  --data datasets/fineweb_10m \
  --dp-size 1 \
  --tp-size 1 \
  --no-use-sp \
  --zero-stage 0 \
  --max-steps 200 \
  --wandb-run-name llama-10m-single
```

Distributed example with TP/SP/ZeRO-3:

```bash
PYTHONPATH=. torchrun --nproc_per_node=4 tools/pretrain.py \
  --config configs/llama_50m.yaml \
  --data datasets/fineweb_50m
```

The script prints a resolved run summary on rank 0, including model size,
mesh, plugins, batch tokens, target tokens, estimated FLOPs/token, logging,
and checkpoint settings.

Use `--dry-run` to validate a recipe without entering the training loop or
initializing W&B:

```bash
PYTHONPATH=. torchrun --nproc_per_node=4 tools/pretrain.py \
  --config configs/llama_50m.yaml \
  --data datasets/fineweb_50m \
  --dry-run
```

Use `--run-manifest` to write the resolved run configuration as JSON. This works
for both dry-runs and normal training runs:

```bash
PYTHONPATH=. torchrun --nproc_per_node=4 tools/pretrain.py \
  --config configs/llama_50m.yaml \
  --data datasets/fineweb_50m \
  --dry-run \
  --run-manifest logs/llama_50m_manifest.json
```

The training script logs `loss`, `lr`, `train/tokens`, `train/tokens_per_sec`,
`perf/step_sec`, `perf/step_sec_window`, estimated `perf/tflops_per_gpu`, and
CUDA memory metrics when CUDA is available. Timing metrics ending in `_sec` are
reported as per-optimizer-step averages over the logging interval; the matching
`*_sec_window` metric records the total wall time for that interval. W&B is
initialized only on rank 0. Fine-grained profiling is intentionally kept out of
the steady-state training path. MFU is a reporting layer concern and can be
computed offline from `perf/tflops_per_gpu` and a declared hardware peak.

Use PyTorch profiler for trace-based performance debugging:

```bash
PYTHONPATH=. torchrun --nproc_per_node=4 tools/pretrain.py \
  --config configs/llama_50m.yaml \
  --data datasets/fineweb_50m \
  --max-steps 20 \
  --torch-profiler \
  --torch-profiler-dir traces/llama_50m \
  --torch-profiler-wait 2 \
  --torch-profiler-warmup 2 \
  --torch-profiler-active 4
```

Profiler traces are written per rank under `rank_XXXXX/` directories. This mode
is for CUDA/NCCL/operator timeline analysis and has non-trivial overhead; keep
it off for normal throughput runs.

Training recipes support AdamW hyperparameters plus constant, linear, and cosine LR schedules:

```yaml
training:
  lr: 3.0e-4
  weight_decay: 0.1
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_eps: 1.0e-8
  fused_adamw: true
  lr_schedule: cosine
  warmup_steps: 100
  min_lr: 3.0e-5
```

The LLaMA path supports block-level activation checkpointing:

```bash
PYTHONPATH=. .venv/bin/python tools/pretrain.py \
  --config configs/llama_50m.yaml \
  --attention-backend sdpa_auto \
  --activation-checkpointing \
  --activation-checkpoint-every-n-layers 2
```

To continue logging into an existing W&B run, pass its run id:

```bash
PYTHONPATH=. torchrun --nproc_per_node=4 tools/pretrain.py \
  --config configs/llama_50m.yaml \
  --data datasets/fineweb_50m \
  --resume-from checkpoints/llama_50m/step_00002500 \
  --wandb-run-id oxqveqbo
```

W&B checkpoint artifacts can be enabled by setting `--wandb-checkpoint-every N`.
`N` must be a multiple of `--checkpoint-every`; local checkpointing remains the
source of truth, and rank 0 uploads selected checkpoint directories
asynchronously as W&B Artifacts.

Checkpoint writes are atomic at the step-directory level: the runtime writes a
`step_XXXXXXXX.tmp` directory first and renames it only after all rank-local
artifacts and the manifest are complete. Recipes can also set retention and
free-space guardrails:

```yaml
checkpoint:
  every: 100
  keep_last: 1
  keep_every_n_steps: 500
  min_free_gb: 5
```

`min_free_gb` is fail-fast: if the checkpoint filesystem has less free space
than requested, training raises instead of writing a partial checkpoint.

Existing checkpoints can also be uploaded manually:

```bash
PYTHONPATH=. .venv/bin/python tools/upload_wandb_checkpoint.py \
  --checkpoint-dir checkpoints/llama_50m_dp2_tp2_sp_zero3 \
  --steps 500 1000 1500 2000 2500 \
  --project maltos \
  --entity xing7-org \
  --artifact-prefix llama-50m-dp2-tp2-sp-zero3-main
```

## Checkpointing

Each checkpoint step is a directory:

```text
checkpoints/tiny/step_00000100/
  manifest.json
  model_rank_0.pt
  optim_rank_0.pt
  trainer_rank_0.pt
  ...
```

The manifest records rank-local model shards, optimizer source ranks, and artifact locations. `StateManager` owns export/import of model, optimizer, trainer, plugin, RNG, and dataloader state.

Resume:

```bash
PYTHONPATH=. .venv/bin/python tools/pretrain.py \
  --data datasets/fineweb_500m \
  --resume-from checkpoints/tiny/step_00000100 \
  --max-steps 200
```

## Design Notes

- The model stays close to normal PyTorch. Parallel behavior is declared by `parallelize_spec()` and applied by runtime plugins.
- `RuntimeCore` is the execution engine. It does not own the dataloader or logging sinks.
- `Trainer` owns the training loop, dataloader binding, checkpoint cadence, and metric cadence.
- Plugins can own optimizers, as ZeRO does. Otherwise `RuntimeCore` owns the optimizer.
- Metrics are produced locally by runtime/plugins, reduced over time by `MetricAggregator`, then optionally reduced across ranks.
- Checkpoint metadata is extensible: plugins can annotate parameter states and export plugin-specific state.

## Current Limitations

- Pipeline parallel, context parallel, and expert parallel are planned but not implemented yet.
- Activation checkpointing is implemented for the LLaMA path; tiny models keep the simpler eager path.
- The LLaMA path supports `eager`, `sdpa_auto`, and `sdpa_flash` attention backends. `sdpa_auto` uses PyTorch's SDPA dispatcher by default; `sdpa_flash` asks PyTorch to use its FlashAttention backend when the shape, dtype, and device are eligible. Custom FlashAttention kernels are not implemented yet.
- The current implementation prioritizes clarity and correctness over Megatron-level throughput optimization.
- The tiny transformer is intentionally small and readable; the LLaMA path is more realistic but still minimal.
- YAML recipes cover the main experiment settings; CLI flags can override any recipe field for quick sweeps.
