# Training System Skeleton

## Layout
- `train_system/parallel`: declarative parallel plan and schedules
- `train_system/runtime`: runtime context and plugin lifecycle hooks
- `train_system/state`: parameter shard metadata and checkpoint spec
- `train_system/train`: trainer loop that is parallel-strategy agnostic
- `train_system/models`: tiny model definitions for local smoke tests and training recipes

## Design Intent
- Keep model code close to single-device semantics.
- Compose parallelism through runtime plugins (`DP/TP/PP/CP/EP/ZeRO`).
- Keep `torch.nn.Parameter` standard; layer ZeRO metadata in `ParamHandle`.

## Next Implementation Steps
1. Add real pretraining entrypoints on top of `Trainer`.
2. Add model variants beyond `TinyTransformer`.
3. Add `PPScheduler` adapters (`1F1B`, interleaved, zero bubble).
4. Add context parallel and expert parallel plugins.

## Local Smoke
```bash
PYTHONPATH=. .venv/bin/python train_system/tests/run_matrix.sh
```
