# Training System Skeleton

## Layout
- `train_system/parallel`: declarative parallel plan (`ProcessMesh`, `ParallelConfig`, schedules)
- `train_system/runtime`: runtime context and plugin lifecycle hooks
- `train_system/state`: parameter shard metadata and checkpoint spec
- `train_system/engine`: trainer loop that is parallel-strategy agnostic
- `train_system/examples`: tiny model for local smoke tests

## Design Intent
- Keep model code close to single-device semantics.
- Compose parallelism through runtime plugins (`DP/TP/PP/CP/EP/ZeRO`).
- Keep `torch.nn.Parameter` standard; layer ZeRO metadata in `ParamHandle`.

## Next Implementation Steps
1. Add `DDPPlugin` with bucket and comm hook controls.
2. Add `TPPlugin` + sequence parallel layout helpers.
3. Add `PPScheduler` adapters (`1F1B`, interleaved, zero bubble).
4. Add `ZeroPlugin` stage 1 -> 2 -> 3 on top of `ParamHandle` and flat buffers.
5. Add sharded checkpoint IO using `CheckpointSpec`.

## Local Smoke
```bash
torchrun --standalone --nproc_per_node=2 tools/smoke_gloo_ddp.py
```
