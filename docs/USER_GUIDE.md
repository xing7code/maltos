# MALTOS User Guide

This guide is for the practical question:

How do I take a user-defined model, choose a runtime topology and plugins, and
train it with MALTOS?

The short answer is:

- If you want to train one of the built-in recipe models, use `tools/pretrain.py`.
- If you want to train your own model today, use `RuntimeCore + Trainer` directly.
- If you want TP/SP/PP/CP/EP support for your own model, you need to expose the
  corresponding model-side specs that MALTOS plugins consume.

## 1. Choose The Right Entry Path

MALTOS currently has two user-facing entry paths.

### Path A: Use the built-in pretraining CLI

Use `tools/pretrain.py` when:

- your model is one of the built-in types
- you want the current token-shard pretraining flow
- you want YAML config, dry-run, checkpointing, and W&B without writing code

Today the CLI only builds:

- `tiny`
- `llama`

So this path is not yet a generic "register any user model" interface.

### Path B: Use `RuntimeCore + Trainer` directly

Use this when:

- you have your own `nn.Module`
- you want a custom dataloader
- you want to control plugin composition in Python
- you are prototyping new training objectives or new model architectures

This is the current recommended path for integrating a custom model.

## 2. Minimum Contract For A Custom Model

At minimum, your model must be a normal `torch.nn.Module` whose `forward()`
returns a scalar training loss tensor for the current batch.

The simplest case is:

```python
import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(128, 1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.proj(batch).pow(2).mean()
```

That is enough to use:

- `RuntimeCore`
- `Trainer`
- checkpointing
- metrics
- precision
- grad clipping
- DDP / ZeRO

without any special model-side parallel spec.

## 3. The Smallest Training Script

This is the minimum end-to-end MALTOS path for a user-defined model.

```python
from __future__ import annotations

import torch
import torch.nn as nn

from data import SimpleTensorDataLoader
from parallel import ParallelPlan
from runtime import MeshConfig, RuntimeCore
from runtime.plugins.grad_clip import GradClipPlugin
from runtime.plugins.metrics import MetricPlugin
from runtime.plugins.precision import PrecisionPlugin
from train import Trainer, TrainerConfig


class MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(128, 1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.proj(batch).pow(2).mean()


def main() -> None:
    data = torch.randn(64, 128)
    dataloader = SimpleTensorDataLoader(data, batch_size=8)

    runtime = RuntimeCore(
        model=MyModel(),
        mesh=MeshConfig(),
        plan=ParallelPlan(),
        optimizer_factory=lambda params: torch.optim.AdamW(params, lr=3e-4),
        scheduler_factory=None,
        plugins=[
            PrecisionPlugin(compute_dtype=torch.bfloat16),
            GradClipPlugin(max_norm=1.0),
            MetricPlugin(),
        ],
    )

    trainer = Trainer(
        runtime=runtime,
        dataloader=dataloader,
        config=TrainerConfig(
            max_steps=100,
            log_every=10,
        ),
    )
    trainer.setup()
    trainer.fit()


if __name__ == "__main__":
    main()
```

That is the core MALTOS integration pattern.

## 4. What `RuntimeCore` Owns vs What `Trainer` Owns

This split matters when you integrate a real model.

`RuntimeCore` owns:

- model execution
- runtime phase hooks
- plugin binding and model transformation
- optimizer creation, unless a plugin owns the optimizer
- scheduler creation, unless a plugin owns the optimizer
- runtime-local execution state

`Trainer` owns:

- dataloader iteration
- optimizer-step cadence
- logging cadence
- checkpoint cadence
- resume path

The normal control flow is:

```python
loss, should_step = runtime.run_step(batch)
if should_step:
    runtime.step_optimizer()
```

You usually do not call that loop manually unless you are building a custom
trainer. Most users should let `Trainer.fit()` drive it.

## 5. Batch Contract

MALTOS does not force one universal batch schema, but the built-in pretraining
path uses:

```python
{
    "input_ids": Tensor[batch, seq],
    "labels": Tensor[batch, seq],
}
```

If your model uses a different batch type, that is fine as long as:

- your dataloader returns it
- your model knows how to consume it
- any active plugin that inspects the batch also supports it

The simplest path for custom work is:

- tensor batch for toy / synthetic tasks
- dict batch for language-model training

## 6. Dataloaders

You have two built-in examples to follow.

### `SimpleTensorDataLoader`

Use this for:

- smoke tests
- synthetic data
- simple custom objectives

It batches a tensor dataset and exposes checkpointable cursor state.

### `PretrainingDataLoader`

Use this for:

- token-shard next-token prediction
- DP-aware deterministic token streaming
- real pretraining runs with resumable shard position

It expects raw `.bin` token shards through `TokenShardDataset`.

## 7. Optimizer And Scheduler Contract

MALTOS uses factories, not prebuilt optimizer objects.

Pass:

- `optimizer_factory`
- `scheduler_factory`

Do not pass a prebuilt optimizer into the runtime.

Example:

```python
runtime = RuntimeCore(
    model=model,
    mesh=MeshConfig(),
    plan=ParallelPlan(),
    optimizer_factory=lambda params: torch.optim.AdamW(params, lr=3e-4),
    scheduler_factory=lambda optim: torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=1000),
)
```

Why this matters:

- plugins may transform or shard the model first
- runtime-owned optimizers are created after model transformation
- ZeRO-style plugins can create optimizer state on the correct sharded parameter objects

## 8. Choosing Plugins

The most common plugin combinations look like this.

### Single-process baseline

```python
plugins = [
    PrecisionPlugin(compute_dtype=torch.bfloat16),
    GradClipPlugin(max_norm=1.0),
    MetricPlugin(),
]
```

### DDP

```python
from runtime.plugins.ddp import DataParallelPlugin

plugins = [
    DataParallelPlugin(async_op=False),
    PrecisionPlugin(compute_dtype=torch.bfloat16),
    GradClipPlugin(max_norm=1.0),
    MetricPlugin(),
]
```

### Bucketed DDP

```python
from runtime.plugins.ddp import BucketDataParallelPlugin

plugins = [
    BucketDataParallelPlugin(bucket_mb_size=25),
    PrecisionPlugin(compute_dtype=torch.bfloat16),
    GradClipPlugin(max_norm=1.0),
    MetricPlugin(),
]
```

### ZeRO

```python
from runtime.plugins.zero3 import Zero3Plugin

plugins = [
    Zero3Plugin(wrap_cls={nn.Linear}),
    PrecisionPlugin(compute_dtype=torch.bfloat16),
    GradClipPlugin(max_norm=1.0),
    MetricPlugin(),
]
```

The runtime resolves plugin order automatically. You provide the set of plugins;
MALTOS applies dependency and ordering constraints internally.

## 9. Mesh And Plan

`MeshConfig` describes topology:

```python
mesh = MeshConfig(
    dp=2,
    tp=2,
    pp=1,
    cp=1,
    ep=1,
)
```

`ParallelPlan` describes strategy choices on top of that topology:

```python
from parallel import ParallelPlan
from parallel.context_interfaces import ContextParallelAttentionCoreType
from parallel.plan import PipelineScheduleConfig

plan = ParallelPlan(
    cp_attn_core=ContextParallelAttentionCoreType.ALL_GATHER_KV,
    pp_schedule=PipelineScheduleConfig(microbatches=4),
)
```

Think of it this way:

- `MeshConfig` says how many ranks live on each axis
- `ParallelPlan` says what algorithm choices you want on those axes

## 10. When A Model Needs Parallel Specs

For plain single-process, DDP, precision, grad clip, and most ZeRO-only usage,
your model can stay an ordinary `nn.Module`.

You only need extra model-side methods when you want plugins that transform or
partition the model structurally.

### TP / SP

Expose:

```python
def tpsp_parallelize_spec(self) -> TpSpParallelSpec: ...
```

This tells TP/SP which module paths to shard and what communication pattern to
use.

### PP

Expose:

```python
def pipeline_parallel_spec(self) -> PipelineParallelSpec: ...
```

This tells PP which modules belong to:

- head
- pipe body
- tail

### CP

Expose:

```python
def context_parallel_spec(self) -> ContextParallelSpec: ...
```

This tells CP which attention modules should receive a CP attention core.

### EP

Expose:

```python
def expert_parallel_spec(self) -> ExpertParallelSpec: ...
```

This tells EP which MoE modules should be expert-sharded.

## 11. Example: Add TP/SP Spec To A Custom Model

The easiest pattern is to copy the built-in TinyTransformer or LLaMA style.

Example:

```python
from parallel.specs import TpSpParallelSpec, TpSpShardAxis, TpSpShardRule


class MyTransformerTp(nn.Module):
    ...

    def tpsp_parallelize_spec(self) -> TpSpParallelSpec:
        rules = []
        for i in range(len(self.layers)):
            rules += [
                TpSpShardRule(f"layers.{i}.attn.q_proj", shard_axis=TpSpShardAxis.PARAM_OUT),
                TpSpShardRule(f"layers.{i}.attn.k_proj", shard_axis=TpSpShardAxis.PARAM_OUT),
                TpSpShardRule(f"layers.{i}.attn.v_proj", shard_axis=TpSpShardAxis.PARAM_OUT),
                TpSpShardRule(
                    f"layers.{i}.attn.o_proj",
                    shard_axis=TpSpShardAxis.PARAM_IN,
                    post_comm="all_reduce",
                ),
            ]
        return TpSpParallelSpec(rules=rules, tie_rules=[])
```

Then enable:

```python
plugins = [TensorParallelPlugin()]
mesh = MeshConfig(tp=2)
```

For SP, add sequence-axis rules and include `SequenceParallelPlugin()` as well.

## 12. Example: Add PP Spec To A Custom Decoder

Your model needs to describe where embeddings, transformer blocks, and output
layers live.

```python
from parallel.specs import PipelineParallelSpec


def pipeline_parallel_spec(self) -> PipelineParallelSpec:
    return PipelineParallelSpec(
        head_layers=["embed_tokens"],
        pipe_layers=["layers"],
        tail_layers=["norm", "lm_head"],
    )
```

Then enable:

```python
plugins = [PipelineParallelPlugin()]
mesh = MeshConfig(pp=2)
plan = ParallelPlan(pp_schedule=PipelineScheduleConfig(microbatches=4))
```

For PP-style decoder models, your forward path should be able to:

- start from `input_ids` on the first stage
- start from `hidden_states` on non-first stages
- return hidden states on non-tail stages
- return logits or loss on the tail stage

The built-in `TinyTransformer` and `LlamaForCausalLM` are the best reference
implementations for this pattern.

## 13. Example: Add CP Spec

Expose the attention module paths:

```python
from parallel.specs import ContextParallelSpec


def context_parallel_spec(self) -> ContextParallelSpec:
    return ContextParallelSpec(
        attention_paths=[f"layers.{i}.self_attn" for i in range(len(self.layers))],
    )
```

Important:

- the listed attention modules must expose an `attn_core` field or equivalent
  protocol-compatible call path
- CP today is a v0 path with sequence divisibility assumptions

If you are integrating a fresh model, start from the LLaMA/TinyTransformer CP
pattern instead of designing this from scratch.

## 14. Example: Add EP Spec

Expose the MoE module paths:

```python
from parallel.specs import ExpertParallelSpec


def expert_parallel_spec(self) -> ExpertParallelSpec:
    return ExpertParallelSpec(
        moe_paths=["layers.0.moe", "layers.1.moe"],
    )
```

Your MoE modules must satisfy the expert-parallel protocol shape expected by the
plugin. The built-in `TinyMoETransformer` is the reference pattern here.

Current status:

- EP is exercised in tests
- EP is not yet exposed through `tools/pretrain.py`

So today EP integration is a code-path feature, not a CLI feature.

## 15. Checkpointing And Resume

If you use `Trainer`, checkpointing is configured through `TrainerConfig`.

```python
trainer = Trainer(
    runtime=runtime,
    dataloader=dataloader,
    config=TrainerConfig(
        max_steps=1000,
        log_every=10,
        checkpoint_every=100,
        checkpoint_dir="checkpoints/my_run",
        checkpoint_keep_last=2,
        checkpoint_keep_every_n_steps=500,
        checkpoint_min_free_gb=5,
    ),
)
```

Resume is:

```python
trainer = Trainer(
    runtime=runtime,
    dataloader=dataloader,
    config=TrainerConfig(
        max_steps=2000,
        resume_from="checkpoints/my_run/step_00001000",
    ),
)
```

MALTOS checkpoints include:

- model state
- optimizer state
- scheduler state
- trainer step context
- RNG state
- plugin state
- dataloader state

## 16. Logging And Metrics

By default, `Trainer` does not force any specific logger.

You can pass:

- a single logger
- a list of loggers

Built-in loggers live in `utils.metrics`.

The pretraining CLI already wires:

- console logging
- JSONL logging
- W&B logging

If you are integrating via Python, you can still reuse those logger classes
directly.

## 17. Current CLI Status For User Models

This is important to understand before you try to force everything through YAML.

`tools/pretrain.py` today is:

- a real pretraining app
- good for built-in LLaMA and tiny recipes
- not yet a generic plugin registry + external model loader

If you want your own model to work through the CLI, you currently need to edit
the app code:

- add your model to `_build_model()`
- possibly add any model-specific config args
- possibly extend `_ZERO3_WRAP_CLS` if your architecture needs different ZeRO-3 wrapping targets

So the clean rule is:

- built-in models: use the CLI
- custom models: use Python integration first
- generic external-model CLI support: future package work

## 18. Recommended Integration Order

If you are bringing up a new model, do it in this order.

1. Make the model train in single-process mode with `RuntimeCore + Trainer`.
2. Add `PrecisionPlugin` and `GradClipPlugin`.
3. Add checkpoint/resume and confirm state round-trips.
4. Add DDP or ZeRO.
5. Add TP/SP if the model is transformer-like and you need tensor sharding.
6. Add PP or CP only after the single-device and DP/ZeRO paths are already stable.
7. Add EP only once the MoE module boundary is explicit and testable.

This order matches how the runtime is easiest to debug.

## 19. Best Reference Files In This Repo

If you are wiring your own model, read these first.

- `tests/smoke_runtime_core.py`
  - smallest runtime-only examples
- `tests/smoke_trainer_loop.py`
  - smallest trainer + dataloader examples
- `models/tiny_transformer.py`
  - clean reference for TP/SP/PP/CP-friendly decoder structure
- `models/llama.py`
  - more realistic pretraining model with activation checkpointing and SDPA
- `models/tiny_moe_transformer.py`
  - reference for EP-style MoE structure
- `tools/pretrain.py`
  - reference app wiring for a complete training recipe

## 20. Practical Takeaways

- MALTOS is already a good runtime substrate for custom training code.
- It is not yet a generic "load arbitrary model by config string" package.
- The stable user path today is Python integration around `RuntimeCore + Trainer`.
- Parallel plugins become available as your model exposes the right structural specs.
- If your model looks like the built-in TinyTransformer/LLaMA/TinyMoE patterns,
  integration is straightforward.
