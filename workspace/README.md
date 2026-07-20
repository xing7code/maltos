# Workspace

This directory is for local or machine-specific experiment artifacts.

Use one subdirectory per model/project, for example:

```text
workspace/
  olmo2_13b_sft/
    data/
    checkpoints/
    logical_checkpoints/
    logs/
    hf_cache/
```

Everything under `workspace/` is ignored by git by default, except this README
and an optional `.gitkeep`. Keep large datasets, downloaded model weights,
runtime checkpoints, logical checkpoints, profiler outputs, and local logs here.
