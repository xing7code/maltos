# CLI Reference

## Preparing Token Shards

The runtime dataloader consumes raw `.bin` token shards. To tokenize a
Hugging Face dataset:

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

## Pretraining CLI

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
plugins, training settings, token targets, logging, profiler, and
checkpoint settings.

Use `--dry-run` to validate a recipe without entering the training loop or
initializing W&B:

```bash
PYTHONPATH=. torchrun --nproc_per_node=4 tools/pretrain.py \
  --config configs/llama_50m.yaml \
  --data datasets/fineweb_50m \
  --dry-run
```

Use `--run-manifest` to write a JSON record of the resolved run
configuration, CLI args, and git metadata. This works for both dry-runs and
normal training runs:

```bash
PYTHONPATH=. torchrun --nproc_per_node=4 tools/pretrain.py \
  --config configs/llama_50m.yaml \
  --data datasets/fineweb_50m \
  --dry-run \
  --run-manifest logs/llama_50m_manifest.json
```

The training script logs `loss`, `lr`, `train/tokens`,
`train/tokens_per_sec`, `perf/step_sec`, `perf/step_sec_window`, estimated
`perf/tflops_per_gpu`, and CUDA memory metrics when CUDA is available.
Timing metrics ending in `_sec` are reported as per-optimizer-step averages
over the logging interval; the matching `*_sec_window` metric records the
total wall time for that interval. Fine-grained profiling is intentionally
kept out of the steady-state training path. MFU is a reporting-layer
concern and can be computed offline from `perf/tflops_per_gpu` and a
declared hardware peak.

Use the PyTorch profiler for trace-based performance debugging:

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

Profiler traces are written per rank under `rank_XXXXX/` directories. This
mode is for CUDA/NCCL/operator timeline analysis and has non-trivial
overhead; keep it off for normal throughput runs.

Training recipes support AdamW hyperparameters plus constant, linear, and
cosine LR schedules:

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

W&B checkpoint artifacts can be enabled by setting
`--wandb-checkpoint-every N`. `N` must be a multiple of `--checkpoint-every`;
local checkpointing remains the source of truth, and rank 0 uploads
selected checkpoint directories asynchronously as W&B Artifacts.

Existing checkpoints can also be uploaded manually:

```bash
PYTHONPATH=. .venv/bin/python tools/upload_wandb_checkpoint.py \
  --checkpoint-dir checkpoints/llama_50m_dp2_tp2_sp_zero3 \
  --steps 500 1000 1500 2000 2500 \
  --project maltos \
  --entity xing7-org \
  --artifact-prefix llama-50m-dp2-tp2-sp-zero3-main
```

## Equivalence & Regression Tests

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

That case exercises:

```
PretrainingDataLoader + TP + SP + ZeRO-3 + bf16 + grad clip
+ gradient accumulation + checkpoint save/load + dataloader resume
```

Run the maintained single-feature regression suite:

```bash
PYTHONPATH=. PYTHON_BIN=.venv/bin/python bash tests/run_single_feature.sh
```

Run the maintained full-stack matrix:

```bash
PYTHONPATH=. PYTHON_BIN=.venv/bin/python bash tests/run_matrix.sh
```
