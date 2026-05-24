# Training System Skeleton

## Layout
- `parallel`: declarative parallel plan and schedules
- `runtime`: runtime context and plugin lifecycle hooks
- `state`: parameter shard metadata and checkpoint spec
- `train`: trainer loop that is parallel-strategy agnostic
- `models`: tiny model definitions for local smoke tests and training recipes

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
PYTHONPATH=. PYTHON_BIN=.venv/bin/python bash tests/run_matrix.sh
```
