# MALTOS Architecture

MALTOS is organized around one idea: keep model code close to ordinary
single-device PyTorch semantics, and push distributed training behavior into a
runtime plus composable plugins.

## Core Pieces

- `RuntimeCore`
  - Owns training phase orchestration.
  - Binds plugins, runs phase hooks, and executes one logical training step.
  - Owns the optimizer and scheduler unless a plugin explicitly takes ownership.
- `Trainer`
  - Owns the outer loop.
  - Drives dataloader iteration, optimizer-step cadence, logging cadence, and checkpoint cadence.
  - Binds the dataloader into `StateManager` and handles resume.
- `RuntimePlugin`
  - Declares ordering constraints and optional runtime hooks.
  - Can transform the model, collect metrics, export/import plugin state, or override the step runner.
  - ZeRO-style plugins can own the optimizer after they finish sharding setup.
- `StateManager`
  - Owns logical training state export/import.
  - Tracks parameter metadata plus model, optimizer, scheduler, trainer, RNG, plugin, and dataloader state.
- `ParallelPlan` and `MeshConfig`
  - `MeshConfig` describes process-mesh axes and process groups.
  - `ParallelPlan` describes mesh-dependent strategy choices such as CP attention core and PP schedule.

## Step Contract

`Trainer` calls:

```python
loss, should_step = runtime.run_step(batch)
if should_step:
    runtime.step_optimizer()
```

This split is deliberate.

- `run_step()` performs forward, backward, plugin phases, and gradient accumulation scaling.
- `step_optimizer()` performs optimizer stepping, scheduler stepping, zeroing grads, and post-step hooks.
- The trainer decides when a logical optimizer step happens.

This keeps gradient accumulation, mid-step checkpoint/resume behavior, and
pipeline-style step runners explicit instead of burying them inside an opaque
trainer loop.

## Phase Model

The runtime exposes these training phases (`RuntimePhase`):

- `PRE_STEP_RUNNER`
- `PRE_FORWARD`
- `POST_FORWARD`
- `PRE_BACKWARD`
- `POST_BACKWARD`
- `PRE_STEP`
- `POST_STEP`
- `PRE_SAVE`
- `POST_LOAD`

Plugin initialization has separate lifecycle hooks: `bind()` attaches runtime
state, `transform_model()` lets TP/SP/PP/ZeRO/precision/etc. rewrite the module,
and `annotate_param_layout()` records parameter layout before optimizer state is
built.

Plugins compose by registering phase behavior rather than rewriting the trainer.
This is what lets TP/SP/PP/CP/DDP/ZeRO/precision/clip/profiler/metrics stack on
one core execution path.

## Optimizer Ownership

MALTOS uses a strict optimizer contract.

- Callers do not pass a prebuilt optimizer or scheduler into the runtime.
- Callers pass `optimizer_factory` and `scheduler_factory`.
- If no plugin owns the optimizer, `RuntimeCore` creates it after model transformation.
- If a plugin owns the optimizer, that plugin is responsible for calling the runtime factories after it finishes sharding or bucketing setup.

This avoids the common failure mode where an optimizer is built on the wrong
parameter objects before TP/ZeRO/PP transformations finish.

## Checkpoint Model

Checkpoints are sharded step directories with rank-local artifacts and a global
manifest.

Each step directory contains:

```text
step_00000100/
  manifest.json
  model_rank_0.pt
  optim_rank_0.pt
  trainer_rank_0.pt
  ...
```

The manifest records:

- checkpoint version
- world size
- per-rank parameter metadata
- optimizer source ranks
- artifact paths

`TrainerState` includes:

- runtime step context
- RNG state
- plugin state
- dataloader state
- consumed token count

Checkpoint writes are atomic at the directory level: MALTOS writes
`step_XXXXXXXX.tmp` first and only renames it after all rank-local artifacts and
the manifest are complete.

## Data Path

The pretraining path uses:

- `TokenShardDataset`
  - memory-mapped `.bin` token shards
- `PretrainingDataLoader`
  - deterministic DP-aware next-token batches
  - resumable shard index and token offset state

The batch contract passed into the model is:

```python
{
    "input_ids": Tensor[batch, seq],
    "labels": Tensor[batch, seq],
}
```

Labels are already aligned with logits; the model does not apply an extra
causal shift.

## Current Runtime Surface

Implemented and exercised in the repo:

- sync / async / bucketed DDP
- TP / SP
- PP
- CP
- EP
- ZeRO-1 / ZeRO-2 / ZeRO-3
- bf16 / fp16 precision hooks
- grad clipping
- steady-state perf metrics
- PyTorch profiler traces
- sharded checkpoint save/load
- pretraining dataloader resume

Current practical boundaries:

- PP is intentionally focused on decoder-only TinyTransformer/LLaMA partitioning.
- CP is a v0 implementation with sequence divisibility constraints and some ZeRO coupling in gradient sync paths.
- EP is exercised in tests, but not exposed through the current pretraining CLI.
- The codebase prioritizes clarity and explicit control flow over peak-throughput micro-optimization.

## Verification

The maintained verification story is:

- smoke tests for runtime core, trainer loop, and pretrain CLI
- targeted equivalence tests for TP / PP / CP / EP / ZeRO combinations
- checkpoint/resume tests, including mid-step resume under gradient accumulation
- a maintained `tests/run_single_feature.sh`
- a maintained `tests/run_matrix.sh`
- GitHub Actions smoke plus distributed regression subset

This is not meant to be a full production training platform. It is meant to be
a readable training-system core that demonstrates real distributed-training
reasoning, real checkpoint semantics, and enough validated surface area to grow
into a broader research training stack.
