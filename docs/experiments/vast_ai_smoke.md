# Vast.ai Pretraining Playbook

Goal: run a real-data LLaMA pretraining recipe with observable metrics,
resumable checkpoints, and a clean path from dry-run validation to a 4-GPU
scale-up run.

W&B reference report:

- https://api.wandb.ai/links/xing7-org/f2s88x30

## 1. Setup

```bash
git clone https://github.com/xing7code/maltos.git
cd maltos
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Optional W&B:

```bash
wandb login
```

## 2. Prepare Token Shards

Prepare a 10M-token smoke dataset:

```bash
PYTHONPATH=. python tools/prepare_token_shards.py \
  --dataset HuggingFaceFW/fineweb-edu \
  --config sample-10BT \
  --split train \
  --column text \
  --tokenizer-name-or-path NousResearch/Llama-2-7b-hf \
  --expected-vocab-size 32000 \
  --output-dir datasets/fineweb_10m \
  --max-tokens 10000000 \
  --tokens-per-shard 5000000 \
  --log-every-tokens 1000000 \
  --streaming
```

Prepare a 50M-token scale-up dataset:

```bash
PYTHONPATH=. python tools/prepare_token_shards.py \
  --dataset HuggingFaceFW/fineweb-edu \
  --config sample-10BT \
  --split train \
  --column text \
  --tokenizer-name-or-path NousResearch/Llama-2-7b-hf \
  --expected-vocab-size 32000 \
  --output-dir datasets/fineweb_50m \
  --max-tokens 50000000 \
  --tokens-per-shard 10000000 \
  --log-every-tokens 1000000 \
  --streaming
```

## 3. Dry-Run First

Validate the recipe before burning GPU time:

```bash
PYTHONPATH=. torchrun --nproc_per_node=4 tools/pretrain.py \
  --config configs/llama_50m.yaml \
  --data datasets/fineweb_50m \
  --dry-run \
  --run-manifest logs/llama_50m_dry_run_manifest.json
```

This builds:

- model
- runtime
- dataloader
- trainer setup path

And then exits before:

- entering `trainer.fit()`
- initializing W&B

Use this step to catch mesh/config/plugin mismatches early.

## 4. Single-GPU Smoke

Override the distributed defaults from `configs/llama_10m.yaml` and run a short
single-GPU smoke:

```bash
PYTHONPATH=. python tools/pretrain.py \
  --config configs/llama_10m.yaml \
  --data datasets/fineweb_10m \
  --dp-size 1 \
  --tp-size 1 \
  --no-use-sp \
  --zero-stage 0 \
  --max-steps 200 \
  --log-every 10 \
  --checkpoint-dir checkpoints/llama_10m_single \
  --metrics-jsonl logs/llama_10m_single.jsonl \
  --wandb-run-name llama-10m-single
```

## 5. Resume Check

Verify checkpoint restore before scaling up:

```bash
PYTHONPATH=. python tools/pretrain.py \
  --config configs/llama_10m.yaml \
  --data datasets/fineweb_10m \
  --dp-size 1 \
  --tp-size 1 \
  --no-use-sp \
  --zero-stage 0 \
  --max-steps 300 \
  --resume-from checkpoints/llama_10m_single/step_00000100 \
  --log-every 10 \
  --metrics-jsonl logs/llama_10m_single_resume.jsonl \
  --wandb-run-name llama-10m-single-resume
```

## 6. 4-GPU Main Run

Run the maintained 4-GPU TP/SP/ZeRO-3 recipe:

```bash
PYTHONPATH=. torchrun --nproc_per_node=4 tools/pretrain.py \
  --config configs/llama_50m.yaml \
  --data datasets/fineweb_50m \
  --checkpoint-dir checkpoints/llama_50m \
  --metrics-jsonl logs/llama_50m.jsonl \
  --wandb-run-name llama-50m-dp2-tp2-sp-zero3
```

This recipe resolves to:

- `DP=2`
- `TP=2`
- `SP=on`
- `ZeRO-3`
- `bf16`
- `grad_clip=1.0`
- `grad_accum_steps=8`

Checkpoint cadence and retention come from `configs/llama_50m.yaml` unless you
override them on the CLI.

## 7. Optional Profiling Pass

Use PyTorch profiler only for short debugging runs:

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

Profiler traces are written per rank and are meant for CUDA/NCCL/operator
timeline inspection, not steady-state throughput measurement.

## Metrics To Watch

- `loss`
- `lr`
- `train/tokens`
- `train/tokens_per_sec`
- `perf/step_sec`
- `perf/step_sec_window`
- `perf/tflops_per_gpu`
- `grad_clip/grad_norm`
- `memory/max_allocated_gb`
- `precision/overflow`

MFU is intentionally not computed in the training path. Interpret
`perf/tflops_per_gpu` offline against the theoretical hardware peak you want to
use in the report.

If you enable W&B checkpoint artifacts:

- set `--wandb-checkpoint-every N`
- ensure `N` is a multiple of `--checkpoint-every`
- remember local checkpointing remains the source of truth

## Expected Real-Run Signals

The maintained 4x4090 50M-token run has already been exercised with the main
recipe family. A representative run reached:

- final step around `3100`
- final loss around `0.56`
- global throughput around `4.2k tokens/sec`
- reserved memory around `1.3 GB / GPU`

Treat those values as a sanity reference, not a hard golden target. Exact
numbers will move with host quality, CUDA stack, shard layout, and background
load on the rented box.

## Practical Notes

- Run `--dry-run` before every new rented-machine recipe.
- Confirm checkpoint restore on a short smoke before longer runs.
- Keep profiler off for throughput measurements.
- Prefer the YAML recipe as the base and use CLI flags only for focused overrides.
