# OLMo2 13B SFT on 8xA100 80GB

This is the budget-conscious MALTOS validation run: start from the public
OLMo2 13B **base** weights, train the public SFT recipe for a short segment,
and check that loss, throughput, checkpointing, and sharded reload are stable.

It is **not** an exact continuation of AI2's original SFT run. AI2 does not
publish an SFT middle checkpoint with its optimizer and dataloader state, and
MALTOS logical-to-runtime conversion deliberately emits a weights-only runtime
checkpoint.

The defaults in `configs/olmo2_13b_sft.yaml` align with the public
Open Instruct recipe where MALTOS has an equivalent setting:

- base model: `allenai/OLMo-2-1124-13B` at `main`
- data: `allenai/tulu-3-sft-olmo-2-mixture-0225`
- seed 8, BF16, LR `5e-6`, linear decay, 3% warmup, no weight decay
- 8 GPUs: TP=2, DP=4, SP, ZeRO-3, global batch 64

MALTOS uses packed fixed-length SFT data and `sdpa_auto`, so neither the exact
batch composition nor the attention kernel is asserted to equal Open
Instruct's. The goal is stable SFT convergence, not bitwise reproduction.

## 0. Set Paths

Run from the repository root on the Vast instance:

```bash
export PROJECT_DIR=workspace/olmo2_13b_sft
export CONFIG=configs/olmo2_13b_sft.yaml
export BASE_REPO=allenai/OLMo-2-1124-13B
export BASE_REVISION=main
export BASE_LOGICAL=$PROJECT_DIR/logical_checkpoints/olmo2_13b_base_main
export BASE_RUNTIME=$PROJECT_DIR/checkpoints/runtime_from_base
export TRAIN_RUNTIME=$PROJECT_DIR/checkpoints/sft_from_base
export LOG_DIR=$PROJECT_DIR/logs
export MASTER_PORT=29611
export PYTHONPATH=.

mkdir -p "$PROJECT_DIR"/data "$PROJECT_DIR"/checkpoints \
  "$PROJECT_DIR"/logical_checkpoints "$LOG_DIR"
```

## 1. Machine and Disk Preflight

Use an 8xA100 **80GB** instance. The target layout is TP=2 and DP=4; 40GB
cards are not the target configuration. Choose a host with at least **512GB
system RAM** (1TB preferred) as well as the eight GPUs: model construction and
logical-to-runtime conversion temporarily use CPU memory before the TP/ZeRO-3
layout is fully materialized. Reserve at least 1TB of local disk: base weights,
prepared data, and two ZeRO-3 training checkpoints with optimizer state must
coexist.

```bash
nvidia-smi
df -h .
free -h

.venv/bin/python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index), torch.cuda.get_device_properties(index).total_memory // 2**30, "GiB")
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 8
PY

PYTHONPATH=. .venv/bin/python - <<'PY'
from utils.attention_backend import AttentionBackend
print("attention backend:", AttentionBackend.SDPA_AUTO)
PY
```

Do not continue if `free -h` shows substantially less than 512GB total RAM or
if the filesystem does not have the requested free space. GPU VRAM sharding
does not eliminate this one-time CPU-side construction/conversion requirement.

`sdpa_auto` is the safe default and lets PyTorch choose its available CUDA SDPA
kernel. Do not set `flash_attn` unless the optional `flash-attn` package has
been installed and verified on this exact CUDA/PyTorch image.

Before either W&B-enabled training command below, authenticate metrics logging
once. No checkpoint artifact upload is configured or enabled.

```bash
wandb login
```

## 2. Prepare the Exact SFT Data Snapshot

The data preparation order is deterministic for the configured seed. Keep the
generated `meta.json`; it records all preprocessing choices.

```bash
test -f "$PROJECT_DIR/data/olmo2_13b_sft/meta.json" || \
  sh tools/sft_data.sh olmo2_13b_sft 2>&1 | tee "$LOG_DIR/prepare_sft_data.log"

test -f "$PROJECT_DIR/data/olmo2_13b_sft/meta.json"
PYTHONPATH=. .venv/bin/python -m json.tool \
  "$PROJECT_DIR/data/olmo2_13b_sft/meta.json" | sed -n '1,160p'
```

## 3. Download and Validate the Public Base Checkpoint

Use the base model, not `OLMo-2-1124-13B-SFT`. The base repo is published in
the `model.safetensors.index.json` format accepted by MALTOS.

```bash
hf download "$BASE_REPO" \
  --revision "$BASE_REVISION" \
  --local-dir "$BASE_LOGICAL"

test -f "$BASE_LOGICAL/model.safetensors.index.json"
du -sh "$BASE_LOGICAL"
```

## 4. Convert Base Logical Weights to an 8-way Runtime Checkpoint

This conversion shards weights for the target TP=2, DP=4, ZeRO-3 runtime. Its
output is intentionally weights-only; the first training segment creates new
optimizer, scheduler, RNG, and dataloader state. The converter reads logical
safetensors on demand per target rank instead of materializing the full logical
checkpoint in every rank's host memory.

```bash
torchrun --nproc_per_node=8 \
  --master_addr 127.0.0.1 \
  --master_port "$MASTER_PORT" \
  tools/convert_checkpoint.py logical-to-runtime \
  --config "$CONFIG" \
  --checkpoint "$BASE_LOGICAL" \
  --output "$BASE_RUNTIME" \
  -- \
  --backend nccl \
  --master-addr 127.0.0.1 \
  --master-port "$MASTER_PORT" \
  2>&1 | tee "$LOG_DIR/base_logical_to_runtime.log"

test -f "$BASE_RUNTIME/runtime_spec.json"
test -f "$BASE_RUNTIME/step_00000000/manifest.json"
```

## 5. Dry-Run the Full 8-GPU Topology

This verifies the data path, model construction, 8-GPU mesh, and checkpoint load
without entering the training loop.

```bash
torchrun --nproc_per_node=8 \
  --master_addr 127.0.0.1 \
  --master_port "$MASTER_PORT" \
  train/cli.py \
  --config "$CONFIG" \
  --resume-from "$BASE_RUNTIME/step_00000000" \
  --load-weights-only \
  --dry-run \
  --wandb-mode disabled \
  --backend nccl \
  2>&1 | tee "$LOG_DIR/dry_run.log"
```

## 6. 10-Step Smoke: Memory, NCCL, and Checkpoint

Run this before the longer jobs. It both proves a real optimizer step and
creates a MALTOS-native runtime checkpoint.

```bash
torchrun --nproc_per_node=8 \
  --master_addr 127.0.0.1 \
  --master_port "$MASTER_PORT" \
  train/cli.py \
  --config "$CONFIG" \
  --resume-from "$BASE_RUNTIME/step_00000000" \
  --load-weights-only \
  --max-steps 10 \
  --warmup-steps 0 \
  --checkpoint-every 10 \
  --checkpoint-dir "$TRAIN_RUNTIME/smoke10" \
  --metrics-jsonl "$LOG_DIR/smoke10_metrics.jsonl" \
  --wandb-mode disabled \
  --backend nccl \
  2>&1 | tee "$LOG_DIR/smoke10.log"

test -f "$TRAIN_RUNTIME/smoke10/step_00000010/manifest.json"
```

If this fails from CUDA OOM, stop here. Do not increase micro-batch size;
reduce sequence length only as a temporary systems smoke and record that it is
no longer the target SFT recipe.

## 6a. Short All-Rank Efficiency Trace

Run profiling as a separate short job. The profiler records CPU, CUDA, and NCCL
events for all eight ranks and writes Chrome/TensorBoard-compatible traces.
First run 40 optimizer steps without recording, then give the profiler five
warmup steps and capture five active steps. This gets past CUDA allocator,
NCCL communicator, dataloader, and LR-warmup transients. Each optimizer step
contains 16 accumulated microsteps, so the capture is still deliberately
bounded.

Do **not** use this run's throughput as the headline throughput number; profiler
instrumentation adds overhead. Use the unprofiled 100- or 500-step run for
`perf/tokens_per_sec`, `perf/step_sec`, and `perf/tflops_per_gpu`.

```bash
torchrun --nproc_per_node=8 \
  --master_addr 127.0.0.1 \
  --master_port "$MASTER_PORT" \
  train/cli.py \
  --config "$CONFIG" \
  --resume-from "$BASE_RUNTIME/step_00000000" \
  --load-weights-only \
  --max-steps 50 \
  --warmup-steps 2 \
  --checkpoint-every 10 \
  --checkpoint-dir "$TRAIN_RUNTIME/profile50" \
  --metrics-jsonl "$LOG_DIR/profile50_metrics.jsonl" \
  --wandb-mode disabled \
  --torch-profiler \
  --torch-profiler-dir "$PROJECT_DIR/traces/profile50" \
  --torch-profiler-wait 40 \
  --torch-profiler-warmup 5 \
  --torch-profiler-active 5 \
  --torch-profiler-repeat 1 \
  --backend nccl \
  2>&1 | tee "$LOG_DIR/profile50.log"

find "$PROJECT_DIR/traces/profile50" -type f | sort
du -sh "$PROJECT_DIR/traces/profile50"
```

Keep `--no-torch-profiler-rank0-only` (the default) for the first trace so
NCCL imbalance across ranks is visible. For a smaller operator-only trace, add
`--torch-profiler-rank0-only`. Leave `record-shapes`, `profile-memory`, and
`with-stack` off initially; each materially inflates trace size and overhead.

Open the exported `*.pt.trace.json` files in
[Perfetto](https://ui.perfetto.dev/) or a local TensorBoard installation. Look
for long idle gaps, uneven rank timelines, and NCCL collectives that dominate
the optimizer-step boundary.

### Archive and Download the Trace

Do not add the raw traces to Git or upload them as W&B artifacts: eight-rank
traces can be hundreds of MB or multiple GB. Archive them on Vast, then pull
them to the local machine with `rsync`, which is resumable if the connection
drops.

On the Vast instance, from the repository root:

```bash
tar -C "$PROJECT_DIR/traces" -czf "$PROJECT_DIR/profile50_traces.tar.gz" profile50
ls -lh "$PROJECT_DIR/profile50_traces.tar.gz"
```

On the local machine, copy the host and port from the Vast **Connect** page and
replace `/path/to/your/vast_key`, `<HOST>`, and `<PORT>` below. The remote path
assumes the repository was cloned at `~/maltos`; change it if your clone lives
elsewhere.

```bash
mkdir -p ~/Downloads/maltos-profile

rsync -avP \
  -e "ssh -i /path/to/your/vast_key -p <PORT>" \
  root@<HOST>:~/maltos/workspace/olmo2_13b_sft/profile50_traces.tar.gz \
  ~/Downloads/maltos-profile/

rsync -avP \
  -e "ssh -i /path/to/your/vast_key -p <PORT>" \
  root@<HOST>:~/maltos/workspace/olmo2_13b_sft/logs/profile50_metrics.jsonl \
  ~/Downloads/maltos-profile/
```

For example, if Vast shows `ssh -p 12345 root@ssh1.vast.ai`, use
`<HOST>=ssh1.vast.ai` and `<PORT>=12345`. `rsync -P` keeps partial files, so
re-run the same command to continue an interrupted download.

Finally, on the local machine:

```bash
cd ~/Downloads/maltos-profile
tar -xzf profile50_traces.tar.gz
```

## 7. 100-Step Stability Run

This is the cheapest meaningful loss-curve check. Three warmup steps round the
recipe's 3% warmup ratio.

```bash
torchrun --nproc_per_node=8 \
  --master_addr 127.0.0.1 \
  --master_port "$MASTER_PORT" \
  train/cli.py \
  --config "$CONFIG" \
  --resume-from "$BASE_RUNTIME/step_00000000" \
  --load-weights-only \
  --max-steps 100 \
  --warmup-steps 3 \
  --checkpoint-every 50 \
  --checkpoint-dir "$TRAIN_RUNTIME/stability100" \
  --metrics-jsonl "$LOG_DIR/stability100_metrics.jsonl" \
  --wandb-project maltos \
  --wandb-run-name olmo2-13b-sft-base-stability100 \
  --backend nccl \
  2>&1 | tee "$LOG_DIR/stability100.log"
```

## 8. 500-Step Main Validation Run

Run only after the 100-step curve is stable. The config defaults already set a
15-step warmup, equal to 3% of 500.

```bash
torchrun --nproc_per_node=8 \
  --master_addr 127.0.0.1 \
  --master_port "$MASTER_PORT" \
  train/cli.py \
  --config "$CONFIG" \
  --resume-from "$BASE_RUNTIME/step_00000000" \
  --load-weights-only \
  --checkpoint-dir "$TRAIN_RUNTIME/main500" \
  --metrics-jsonl "$LOG_DIR/main500_metrics.jsonl" \
  --wandb-project maltos \
  --wandb-run-name olmo2-13b-sft-base-main500 \
  --backend nccl \
  2>&1 | tee "$LOG_DIR/main500.log"
```

## 9. Resume a Native Sharded Runtime Checkpoint

Unlike the public-base conversion, this restores MALTOS model, optimizer,
scheduler, RNG, trainer, plugin, and dataloader state. This is the required
resume check for a real run.

```bash
torchrun --nproc_per_node=8 \
  --master_addr 127.0.0.1 \
  --master_port "$MASTER_PORT" \
  train/cli.py \
  --config "$CONFIG" \
  --resume-from "$TRAIN_RUNTIME/main500/step_00000250" \
  --max-steps 500 \
  --checkpoint-every 10 \
  --checkpoint-dir "$TRAIN_RUNTIME/resume500" \
  --metrics-jsonl "$LOG_DIR/resume500_metrics.jsonl" \
  --wandb-project maltos \
  --wandb-run-name olmo2-13b-sft-base-resume500 \
  --backend nccl \
  2>&1 | tee "$LOG_DIR/resume500.log"
```

## 10. What to Inspect

The main evidence is the rank-0 JSONL/W&B curve. Check that:

- `train/loss` is finite and trends down after warmup;
- `grad_clip/grad_norm` remains finite and does not grow persistently;
- `perf/tokens_per_sec`, `perf/step_sec`, and memory are stable after the
  first several steps;
- `step_00000500/manifest.json` exists, and the native resume from step 250
  reaches step 500 without a loss discontinuity beyond normal batch variation.

This validates 13B SFT convergence and sharded runtime restore. It does not
claim an exact match to AI2's original SFT loss curve.
