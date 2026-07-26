#!/usr/bin/env python3
"""Profile a MALTOS training runtime with deterministic synthetic data.

This is a systems benchmark, not a training-quality test.  It builds the
normal MALTOS model/runtime selected by a training recipe or runtime_spec,
but replaces checkpoint restore with deterministic valid synthetic causal-LM
batches. Packed mode varies segment lengths and supervision masks per
micro-batch; dense mode models ordinary pretraining. The default learning rate
is zero so repeated synthetic optimization cannot drift or diverge while the
complete forward/backward/optimizer path is still exercised.

Run it under torchrun.  A recipe supplies the model and parallel topology;
the optional case file supplies reusable experiment defaults and overrides.

Examples:

  # List reusable experiment definitions.
  PYTHONPATH=. .venv/bin/python tools/profile_train_perf.py --list-cases

  # Recipe directly: 5 warmup global steps plus 10 measured global steps.
  torchrun --nproc_per_node=8 tools/profile_train_perf.py \\
    --recipe configs/olmo2_13b_sft.yaml --warmup 5 --steps 10

  # A named case, with a rank-0 torch.profiler trace.
  torchrun --nproc_per_node=8 tools/profile_train_perf.py \\
    --case olmo2_13b_sft_packed --profile --output-dir profiles/olmo2-packed

  # Override normal recipe flags after ``--`` without copying a YAML recipe.
  torchrun --nproc_per_node=8 tools/profile_train_perf.py \\
    --case olmo2_13b_sft_packed -- --micro-batch-size 2 --grad-accum-steps 8

The recipe's data, checkpoint, W&B, and max-step fields are intentionally not
used.  To benchmark trainable updates instead of a stable synthetic workload,
pass ``--use-recipe-optimizer`` (or ``--benchmark-lr VALUE``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import hashlib
import socket
import subprocess
import sys
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml

# Executing ``python tools/profile_train_perf.py`` puts tools/ ahead of the
# repository root on sys.path, where tools/train.py would otherwise shadow the
# train package used below.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from runtime.core import RuntimeCore
from runtime.mesh import MeshAxis, MeshConfig
from data import PrefetchDataLoader
from data.bytescale_hdp import ByteScaleHdpDataLoader
from parallel.hdp_helper import BYTESCALE_HDP_SCHEDULE_KEY, BYTESCALE_HDP_WAVES_KEY
from parallel.context_batch import ContextParallelBatchSharder
from parallel.context_token_planner import build_context_token_planner
from train import cli as train_cli
from train.flags import build_arg_parser, build_runtime_spec, parse_args_from, parse_runtime_spec_args
from utils.constants import (
    IGNORE_INDEX,
    INPUT_IDS_KEY,
    LABELS_KEY,
    PAD_SEQUENCE_ID,
    POSITION_IDS_KEY,
    SEQUENCE_IDS_KEY,
)


DEFAULT_CASE_FILE = Path("configs/profile_train_perf_cases.yaml")


@dataclass(frozen=True)
class CaseDefinition:
    name: str
    description: str
    recipe: str | None
    runtime_spec: str | None
    recipe_overrides: tuple[str, ...]
    synthetic_layout: str | None
    packed_sequences: int | None
    segment_count_min: int | None
    segment_count_max: int | None
    supervised_fraction_min: float | None
    supervised_fraction_max: float | None
    max_padding_fraction: float | None
    delivery: str | None
    data_source: str | None
    document_length_trace: tuple[tuple[int, ...], ...] | None
    require_calibrated_hdp_cost: bool
    warmup: int | None
    steps: int | None


@dataclass(frozen=True)
class SyntheticBatchConfig:
    """Input-distribution controls for one synthetic benchmark workload."""

    layout: str
    segment_count_min: int
    segment_count_max: int
    supervised_fraction_min: float
    supervised_fraction_max: float
    max_padding_fraction: float
    delivery: str
    document_length_trace: tuple[tuple[int, ...], ...] | None = None


def _parse_tool_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--recipe", type=str, help="normal MALTOS training recipe YAML")
    source.add_argument(
        "--runtime-spec",
        type=str,
        help="runtime_spec.json file or checkpoint directory containing it",
    )
    parser.add_argument("--case-file", type=Path, default=DEFAULT_CASE_FILE)
    parser.add_argument("--case", type=str, help="named experiment from --case-file")
    parser.add_argument("--list-cases", action="store_true", help="print cases and exit")
    parser.add_argument("--warmup", type=int, default=None, help="untimed global optimizer steps")
    parser.add_argument("--steps", type=int, default=None, help="timed global optimizer steps")
    parser.add_argument("--synthetic-layout", choices=("dense", "packed"), default=None)
    parser.add_argument(
        "--packed-input",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="shortcut for --synthetic-layout packed/ dense (dense is pretraining)",
    )
    parser.add_argument(
        "--packed-sequences",
        type=int,
        default=None,
        help="legacy fixed segment count; prefer --segment-count-min/max",
    )
    parser.add_argument("--segment-count-min", type=int, default=None)
    parser.add_argument("--segment-count-max", type=int, default=None)
    parser.add_argument("--supervised-fraction-min", type=float, default=None)
    parser.add_argument("--supervised-fraction-max", type=float, default=None)
    parser.add_argument(
        "--max-padding-fraction",
        type=float,
        default=None,
        help="maximum trailing-pad fraction for packed synthetic rows",
    )
    parser.add_argument(
        "--synthetic-delivery",
        choices=("host", "cuda"),
        default=None,
        help="host simulates normal dataloader-to-runtime delivery; cuda isolates runtime compute",
    )
    parser.add_argument(
        "--data-source",
        choices=("synthetic", "recipe"),
        default=None,
        help="synthetic generator or the recipe's real mmap-backed dataloader",
    )
    parser.add_argument("--seed", type=int, default=1234, help="synthetic data seed")
    parser.add_argument("--output-dir", type=Path, default=Path("profiles/train_perf"))
    parser.add_argument("--profile", action="store_true", help="export a torch.profiler trace")
    parser.add_argument(
        "--profile-all-ranks",
        action="store_true",
        help="trace every rank (large); default traces rank 0 only",
    )
    parser.add_argument("--profile-record-shapes", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--use-recipe-optimizer", action="store_true")
    parser.add_argument(
        "--benchmark-lr",
        type=float,
        default=0.0,
        help="override LR for synthetic benchmarking (default: 0, ignored with --use-recipe-optimizer)",
    )
    parser.add_argument("recipe_overrides", nargs=argparse.REMAINDER, help="normal training flags after --")
    args = parser.parse_args()
    if args.recipe_overrides and args.recipe_overrides[0] == "--":
        args.recipe_overrides = args.recipe_overrides[1:]
    if args.warmup is not None and args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be >= 1")
    if args.packed_sequences is not None and args.packed_sequences < 2:
        parser.error("--packed-sequences must be >= 2")
    for name in ("supervised_fraction_min", "supervised_fraction_max", "max_padding_fraction"):
        value = getattr(args, name)
        if value is not None and not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    return args


def _load_cases(path: Path) -> dict[str, CaseDefinition]:
    if not path.is_file():
        if path == DEFAULT_CASE_FILE:
            return {}
        raise FileNotFoundError(f"case file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict) or not isinstance(raw.get("cases"), dict):
        raise ValueError(f"case file {path} must contain a top-level 'cases' mapping")
    cases: dict[str, CaseDefinition] = {}
    for name, value in raw["cases"].items():
        if not isinstance(name, str) or not isinstance(value, dict):
            raise ValueError("each profile case must be a string name mapped to a YAML mapping")
        recipe = value.get("recipe")
        runtime_spec = value.get("runtime_spec")
        if (recipe is None) == (runtime_spec is None):
            raise ValueError(f"case {name!r} must specify exactly one of recipe or runtime_spec")
        overrides = value.get("recipe_overrides", [])
        synthetic = value.get("synthetic", {})
        if not isinstance(overrides, list) or not all(isinstance(item, str) for item in overrides):
            raise ValueError(f"case {name!r}.recipe_overrides must be a list of CLI strings")
        if not isinstance(synthetic, dict):
            raise ValueError(f"case {name!r}.synthetic must be a mapping")
        packed = synthetic.get("packed")
        if packed is not None and not isinstance(packed, bool):
            raise ValueError(f"case {name!r}.synthetic.packed must be true or false")
        layout = synthetic.get("layout")
        if layout is not None and layout not in {"dense", "packed"}:
            raise ValueError(f"case {name!r}.synthetic.layout must be dense or packed")
        if packed is not None:
            inferred_layout = "packed" if packed else "dense"
            if layout is not None and layout != inferred_layout:
                raise ValueError(f"case {name!r}.synthetic.packed disagrees with synthetic.layout")
            layout = inferred_layout
        cases[name] = CaseDefinition(
            name=name,
            description=str(value.get("description", "")),
            recipe=None if recipe is None else str(recipe),
            runtime_spec=None if runtime_spec is None else str(runtime_spec),
            recipe_overrides=tuple(overrides),
            synthetic_layout=layout,
            packed_sequences=_optional_positive_int(synthetic.get("packed_sequences"), f"case {name}.packed_sequences"),
            segment_count_min=_optional_positive_int(synthetic.get("segment_count_min"), f"case {name}.segment_count_min"),
            segment_count_max=_optional_positive_int(synthetic.get("segment_count_max"), f"case {name}.segment_count_max"),
            supervised_fraction_min=_optional_fraction(synthetic.get("supervised_fraction_min"), f"case {name}.supervised_fraction_min"),
            supervised_fraction_max=_optional_fraction(synthetic.get("supervised_fraction_max"), f"case {name}.supervised_fraction_max"),
            max_padding_fraction=_optional_fraction(synthetic.get("max_padding_fraction"), f"case {name}.max_padding_fraction"),
            delivery=_optional_delivery(synthetic.get("delivery"), f"case {name}.delivery"),
            data_source=_optional_data_source(synthetic.get("data_source"), f"case {name}.data_source"),
            document_length_trace=_optional_document_length_trace(
                synthetic.get("document_length_trace"), f"case {name}.document_length_trace"
            ),
            require_calibrated_hdp_cost=bool(value.get("require_calibrated_hdp_cost", False)),
            warmup=_optional_nonnegative_int(value.get("warmup"), f"case {name}.warmup"),
            steps=_optional_positive_int(value.get("steps"), f"case {name}.steps"),
        )
    return cases


def _optional_positive_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _optional_nonnegative_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _optional_fraction(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (float, int)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{label} must be a number in [0, 1]")
    return float(value)


def _optional_delivery(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if value not in {"host", "cuda"}:
        raise ValueError(f"{label} must be host or cuda")
    return str(value)


def _optional_data_source(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if value not in {"synthetic", "recipe"}:
        raise ValueError(f"{label} must be synthetic or recipe")
    return str(value)


def _optional_document_length_trace(value: Any, label: str) -> tuple[tuple[int, ...], ...] | None:
    """Parse one canonical packed batch as rows of explicit document lengths."""
    if value is None:
        return None
    if not isinstance(value, list) or not value or not all(isinstance(row, list) and row for row in value):
        raise ValueError(f"{label} must be a non-empty list of non-empty length lists")
    trace: list[tuple[int, ...]] = []
    for row in value:
        if not all(isinstance(length, int) and length >= 2 for length in row):
            raise ValueError(f"{label} lengths must be integers >= 2")
        trace.append(tuple(row))
    return tuple(trace)


def _resolve_source(args: argparse.Namespace, cases: dict[str, CaseDefinition]) -> CaseDefinition:
    if args.case is not None:
        if args.recipe is not None or args.runtime_spec is not None:
            raise ValueError("--case cannot be combined with --recipe or --runtime-spec")
        try:
            return cases[args.case]
        except KeyError as exc:
            available = ", ".join(sorted(cases)) or "(none)"
            raise ValueError(f"unknown case {args.case!r}; available: {available}") from exc
    if args.recipe is None and args.runtime_spec is None:
        raise ValueError("provide --recipe, --runtime-spec, or --case")
    return CaseDefinition(
        name="direct",
        description="direct command-line source",
        recipe=args.recipe,
        runtime_spec=args.runtime_spec,
        recipe_overrides=(),
        synthetic_layout=None,
        packed_sequences=None,
        segment_count_min=None,
        segment_count_max=None,
        supervised_fraction_min=None,
        supervised_fraction_max=None,
        max_padding_fraction=None,
        delivery=None,
        data_source=None,
        document_length_trace=None,
        require_calibrated_hdp_cost=False,
        warmup=None,
        steps=None,
    )


def _load_training_args(case: CaseDefinition, command_overrides: list[str]) -> argparse.Namespace:
    overrides = [*case.recipe_overrides, *command_overrides]
    if case.recipe is not None:
        return parse_args_from(["--config", case.recipe, *overrides], require_data=False)

    assert case.runtime_spec is not None
    spec_path = Path(case.runtime_spec)
    if spec_path.is_dir():
        spec_path /= "runtime_spec.json"
    raw = json.loads(spec_path.read_text(encoding="utf-8"))
    runtime_defaults = vars(parse_runtime_spec_args(raw))
    parser = build_arg_parser()
    parser.set_defaults(**runtime_defaults)
    return parser.parse_args(overrides)


def _configure_benchmark_args(
    train_args: argparse.Namespace,
    tool_args: argparse.Namespace,
    case: CaseDefinition,
) -> tuple[SyntheticBatchConfig, str, int, int]:
    if tool_args.packed_input is not None and tool_args.synthetic_layout is not None:
        expected = "packed" if tool_args.packed_input else "dense"
        if tool_args.synthetic_layout != expected:
            raise ValueError("--packed-input disagrees with --synthetic-layout")
    layout = (
        ("packed" if tool_args.packed_input else "dense")
        if tool_args.packed_input is not None
        else tool_args.synthetic_layout or case.synthetic_layout or "packed"
    )
    legacy_segments = tool_args.packed_sequences or case.packed_sequences
    segment_count_min = tool_args.segment_count_min or case.segment_count_min or legacy_segments or 3
    segment_count_max = tool_args.segment_count_max or case.segment_count_max or legacy_segments or 7
    if segment_count_min > segment_count_max:
        raise ValueError("segment_count_min must be <= segment_count_max")
    if segment_count_max > train_args.seq_len:
        raise ValueError(
            f"segment_count_max={segment_count_max} exceeds recipe seq_len={train_args.seq_len}"
        )
    supervised_fraction_min = (
        tool_args.supervised_fraction_min
        if tool_args.supervised_fraction_min is not None
        else case.supervised_fraction_min if case.supervised_fraction_min is not None else 0.55
    )
    supervised_fraction_max = (
        tool_args.supervised_fraction_max
        if tool_args.supervised_fraction_max is not None
        else case.supervised_fraction_max if case.supervised_fraction_max is not None else 0.70
    )
    if supervised_fraction_min > supervised_fraction_max:
        raise ValueError("supervised_fraction_min must be <= supervised_fraction_max")
    max_padding_fraction = (
        tool_args.max_padding_fraction
        if tool_args.max_padding_fraction is not None
        else case.max_padding_fraction if case.max_padding_fraction is not None else 0.02
    )
    delivery = tool_args.synthetic_delivery or case.delivery or "host"
    synthetic = SyntheticBatchConfig(
        layout=layout,
        segment_count_min=segment_count_min,
        segment_count_max=segment_count_max,
        supervised_fraction_min=supervised_fraction_min,
        supervised_fraction_max=supervised_fraction_max,
        max_padding_fraction=max_padding_fraction,
        delivery=delivery,
        document_length_trace=case.document_length_trace,
    )
    if synthetic.document_length_trace is not None:
        if len(synthetic.document_length_trace) != train_args.micro_batch_size:
            raise ValueError(
                "document_length_trace row count must equal the logical micro_batch_size "
                f"({len(synthetic.document_length_trace)} != {train_args.micro_batch_size})"
            )
        if any(sum(row) > train_args.seq_len for row in synthetic.document_length_trace):
            raise ValueError("every document_length_trace row must fit within seq_len")
    warmup = tool_args.warmup if tool_args.warmup is not None else case.warmup
    steps = tool_args.steps if tool_args.steps is not None else case.steps
    warmup = 5 if warmup is None else warmup
    steps = 10 if steps is None else steps
    if not tool_args.use_recipe_optimizer:
        train_args.lr = tool_args.benchmark_lr
        train_args.weight_decay = 0.0
        train_args.lr_schedule = "constant"
        train_args.warmup_steps = 0
        train_args.min_lr = 0.0
    if case.require_calibrated_hdp_cost and (
        train_args.hdp_cost_alpha,
        train_args.hdp_cost_beta,
        train_args.hdp_cost_gamma,
    ) == (1.0, 0.0, 0.0):
        raise ValueError(
            f"paper case {case.name!r} requires calibrated --hdp-cost-alpha/beta/gamma; "
            "the default 1/0/0 preflight proxy is not permitted"
        )
    # The benchmark never calls Trainer, so ensure unrelated side effects are disabled.
    train_args.disable_metrics = True
    train_args.wandb_mode = "disabled"
    train_args.checkpoint_every = None
    train_args.resume_from = None
    train_args.load_weights_only = False
    train_args.max_steps = warmup + steps
    if tool_args.profile:
        train_args.torch_profiler = True
        train_args.torch_profiler_dir = str(tool_args.output_dir / "traces")
        # The plugin counts optimizer steps.  It starts in setup, so skipping
        # our benchmark warmup makes the active window align with measured work.
        train_args.torch_profiler_wait = warmup
        train_args.torch_profiler_warmup = 0
        train_args.torch_profiler_active = steps
        train_args.torch_profiler_repeat = 1
        train_args.torch_profiler_rank0_only = not tool_args.profile_all_ranks
        train_args.torch_profiler_record_shapes = tool_args.profile_record_shapes
        train_args.torch_profiler_profile_memory = tool_args.profile_memory
    else:
        train_args.torch_profiler = False
    data_source = tool_args.data_source or case.data_source or "synthetic"
    return synthetic, data_source, warmup, steps


def make_synthetic_batch(
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    config: SyntheticBatchConfig,
    seed: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Create a deterministic valid causal-LM batch outside the timed region."""
    if batch_size < 1 or seq_len < 2:
        raise ValueError("synthetic batch requires batch_size >= 1 and seq_len >= 2")
    if vocab_size < 2:
        raise ValueError("synthetic batch requires vocab_size >= 2")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, generator=generator)
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = IGNORE_INDEX
    batch: dict[str, torch.Tensor] = {INPUT_IDS_KEY: input_ids, LABELS_KEY: labels}
    if config.layout == "dense":
        return batch
    if config.layout != "packed":
        raise ValueError(f"unsupported synthetic layout={config.layout!r}")
    if not 2 <= config.segment_count_min <= config.segment_count_max <= seq_len:
        raise ValueError(
            f"segment count must be in [2, {seq_len}], got "
            f"[{config.segment_count_min}, {config.segment_count_max}]"
        )

    position_ids = torch.empty_like(input_ids)
    sequence_ids = torch.empty_like(input_ids)
    for row in range(batch_size):
        if config.document_length_trace is not None:
            lengths = torch.tensor(config.document_length_trace[row], dtype=torch.long, device=device)
            active_tokens = int(lengths.sum().item())
            if active_tokens > seq_len:
                raise ValueError("document_length_trace exceeds seq_len")
        else:
            segment_count = int(torch.randint(
                config.segment_count_min, config.segment_count_max + 1, (), device=device, generator=generator
            ).item())
            max_padding = int(seq_len * config.max_padding_fraction)
            padding = int(torch.randint(max_padding + 1, (), device=device, generator=generator).item())
            active_tokens = seq_len - padding
            if active_tokens < 2 * segment_count:
                raise ValueError("padding and segment settings leave fewer than two tokens per segment")
            lengths = _random_segment_lengths(
                active_tokens=active_tokens, segment_count=segment_count, generator=generator, device=device,
            )
        padding = seq_len - active_tokens
        start = 0
        for segment, length in enumerate(lengths.tolist()):
            end = start + length
            target_fraction = _uniform_fraction(
                config.supervised_fraction_min,
                config.supervised_fraction_max,
                generator=generator,
                device=device,
            )
            supervised_targets = torch.rand(length - 1, device=device, generator=generator) < target_fraction
            labels[row, start : end - 1] = torch.where(
                supervised_targets,
                input_ids[row, start + 1 : end],
                torch.full((length - 1,), IGNORE_INDEX, dtype=labels.dtype, device=device),
            )
            labels[row, end - 1] = IGNORE_INDEX
            position_ids[row, start:end] = torch.arange(length, dtype=torch.long, device=device)
            sequence_ids[row, start:end] = row * max(config.segment_count_max, len(lengths)) + segment
            start = end
        if padding:
            input_ids[row, active_tokens:] = 0
            labels[row, active_tokens:] = IGNORE_INDEX
            position_ids[row, active_tokens:] = 0
            sequence_ids[row, active_tokens:] = PAD_SEQUENCE_ID
    batch[POSITION_IDS_KEY] = position_ids.contiguous()
    batch[SEQUENCE_IDS_KEY] = sequence_ids.contiguous()
    return batch


def _random_segment_lengths(
    *,
    active_tokens: int,
    segment_count: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Positive, deliberately uneven packed-example lengths summing to active_tokens."""
    lengths = torch.full((segment_count,), 2, dtype=torch.long, device=device)
    remaining = active_tokens - int(lengths.sum().item())
    if remaining == 0:
        return lengths
    # Random weighted allocation creates the variable segment lengths that the
    # real best-fit SFT packer produces, while retaining deterministic seeds.
    weights = torch.rand(segment_count, device=device, generator=generator)
    additions = torch.floor(weights / weights.sum() * remaining).to(torch.long)
    lengths += additions
    residual = remaining - int(additions.sum().item())
    if residual:
        order = torch.randperm(segment_count, device=device, generator=generator)
        lengths[order[:residual]] += 1
    return lengths


def _uniform_fraction(
    low: float,
    high: float,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> float:
    if low == high:
        return low
    return low + (high - low) * float(torch.rand((), device=device, generator=generator).item())


@dataclass
class SyntheticBatchStream:
    """Deterministic per-microbatch producer with trainer-like host delivery."""

    batch_size: int
    seq_len: int
    vocab_size: int
    config: SyntheticBatchConfig
    seed: int
    runtime_device: torch.device
    _index: int = 0
    cp_batch_sharder: Any = None

    def next_batch(self, cp_rank: int | None = None) -> dict[str, torch.Tensor]:
        batch_device = self.runtime_device if self.config.delivery == "cuda" else torch.device("cpu")
        batch = make_synthetic_batch(
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            vocab_size=self.vocab_size,
            config=self.config,
            seed=self.seed + self._index,
            device=batch_device,
        )
        self._index += 1
        if cp_rank is not None:
            if self.cp_batch_sharder is None:
                raise RuntimeError("CP-local synthetic batches require constructor cp_batch_sharder")
            return self.cp_batch_sharder.shard(batch, rank=cp_rank)
        return batch

    @property
    def consumed_tokens(self) -> int:
        return self._index * self.batch_size * self.seq_len

    def state_dict(self) -> dict[str, int]:
        return {"index": self._index}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self._index = int(state["index"])

    def close(self) -> None:
        return None


@dataclass
class _PrefetchMetrics:
    """Thread-safe, non-step-timed observations from synthetic HDP loading."""

    scheduler_materialization_seconds: float = 0.0
    scheduler_materialization_batches: int = 0
    prefetch_thread_batches: int = 0
    real_rank_waves: int = 0
    dummy_rank_waves: int = 0
    wave_counts: list[int] = field(default_factory=list)
    degree_histogram: dict[int, int] = field(default_factory=dict)
    document_length_histogram: dict[int, int] = field(default_factory=dict)
    active_tokens: int = 0
    document_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def observe_hdp(self, batch: Any, *, elapsed_seconds: float) -> None:
        waves = batch.get(BYTESCALE_HDP_WAVES_KEY) if isinstance(batch, dict) else None
        if waves is None:
            return
        layouts = [wave[BYTESCALE_HDP_SCHEDULE_KEY] for wave in waves]
        with self._lock:
            self.scheduler_materialization_seconds += elapsed_seconds
            self.scheduler_materialization_batches += 1
            self.prefetch_thread_batches += int(threading.current_thread().name.startswith("maltos-data"))
            self.wave_counts.append(len(layouts))
            for layout in layouts:
                if layout.documents:
                    self.real_rank_waves += 1
                else:
                    self.dummy_rank_waves += 1
                # Count a document only on the first ring owner: each owner
                # sees exactly one complete global schedule across all ranks.
                for document in layout.documents:
                    if layout.rank != document.participant_ranks[0]:
                        continue
                    self.document_count += 1
                    self.active_tokens += document.sequence_length
                    degree = len(document.participant_ranks)
                    self.degree_histogram[degree] = self.degree_histogram.get(degree, 0) + 1
                    self.document_length_histogram[document.sequence_length] = (
                        self.document_length_histogram.get(document.sequence_length, 0) + 1
                    )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "scheduler_materialization_seconds": self.scheduler_materialization_seconds,
                "scheduler_materialization_batches": self.scheduler_materialization_batches,
                "prefetch_thread_batches": self.prefetch_thread_batches,
                "real_rank_waves": self.real_rank_waves,
                "dummy_rank_waves": self.dummy_rank_waves,
                "wave_counts": list(self.wave_counts),
                "degree_histogram": dict(self.degree_histogram),
                "document_length_histogram": dict(self.document_length_histogram),
                "active_tokens": self.active_tokens,
                "document_count": self.document_count,
            }


class _TimedHdpSyntheticLoader:
    """Measure HDP scheduling/materialization in the prefetch worker only."""

    def __init__(self, loader: ByteScaleHdpDataLoader, metrics: _PrefetchMetrics) -> None:
        self.loader, self.metrics = loader, metrics

    def next_batch(self, cp_rank: int | None = None) -> Any:
        started = time.perf_counter()
        batch = self.loader.next_batch(cp_rank)
        self.metrics.observe_hdp(batch, elapsed_seconds=time.perf_counter() - started)
        return batch

    @property
    def consumed_tokens(self):
        return self.loader.consumed_tokens

    def state_dict(self):
        return self.loader.state_dict()

    def load_state_dict(self, state):
        return self.loader.load_state_dict(state)

    def close(self) -> None:
        self.loader.close()


def _run_global_step(core: RuntimeCore, batches: Any) -> torch.Tensor:
    last_loss: torch.Tensor | None = None
    for micro_step in range(core.grad_accum_steps):
        cp_rank = None
        if core.plan.batch_data_cp_aware:
            cp_group = core.get_group(MeshAxis.CP)
            assert cp_group is not None
            cp_rank = dist.get_rank(cp_group)
        batch = batches.next_batch(cp_rank)
        loss, should_step = core.run_step(batch)
        if micro_step + 1 == core.grad_accum_steps and not should_step:
            raise RuntimeError("runtime did not request an optimizer step at the accumulation boundary")
        if micro_step + 1 != core.grad_accum_steps and should_step:
            raise RuntimeError("runtime requested optimizer step before accumulation boundary")
        last_loss = loss
    core.step_optimizer()
    assert last_loss is not None
    return last_loss


def _check_finite(loss: torch.Tensor, core: RuntimeCore, *, phase: str, step: int) -> float:
    loss_value = float(loss.detach().float().item())
    if not math.isfinite(loss_value):
        raise FloatingPointError(f"non-finite synthetic loss during {phase} step {step}: {loss_value}")
    grad_norm = core.state.metadata.get("zero3/grad_norm", core.state.metadata.get("grad_norm"))
    if isinstance(grad_norm, (float, int)) and not math.isfinite(float(grad_norm)):
        raise FloatingPointError(f"non-finite gradient norm during {phase} step {step}: {grad_norm}")
    return loss_value


def _timing_summary(local_seconds: list[float], device: torch.device) -> list[float]:
    local = torch.tensor(local_seconds, dtype=torch.float64, device=device)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    # Every optimizer step succeeds only when the slowest rank finishes, so its
    # per-step maximum is the meaningful distributed latency.
    critical = torch.stack(gathered).amax(dim=0)
    return [float(value) for value in critical.cpu().tolist()]


def _aggregate_prefetch_metrics(metrics: _PrefetchMetrics | None) -> dict[str, Any] | None:
    if metrics is None:
        return None
    local = metrics.snapshot()
    snapshots = [local]
    if dist.is_initialized():
        snapshots = [None] * dist.get_world_size()
        dist.all_gather_object(snapshots, local)
    aggregate: dict[str, Any] = {
        "scheduler_materialization_seconds_sum": 0.0,
        "scheduler_materialization_seconds_rank_max": 0.0,
        "scheduler_materialization_batches": 0,
        "prefetch_thread_batches": 0,
        "real_rank_waves": 0,
        "dummy_rank_waves": 0,
        "wave_count_histogram": {},
        "degree_histogram": {},
        "document_length_histogram": {},
        "active_tokens_owner_counted": 0,
        "document_count_owner_counted": 0,
    }
    for item in snapshots:
        assert item is not None
        duration = float(item["scheduler_materialization_seconds"])
        aggregate["scheduler_materialization_seconds_sum"] += duration
        aggregate["scheduler_materialization_seconds_rank_max"] = max(
            aggregate["scheduler_materialization_seconds_rank_max"], duration
        )
        for key in ("scheduler_materialization_batches", "prefetch_thread_batches", "real_rank_waves", "dummy_rank_waves"):
            aggregate[key] += int(item[key])
        aggregate["active_tokens_owner_counted"] += int(item["active_tokens"])
        aggregate["document_count_owner_counted"] += int(item["document_count"])
        for source, target in (
            (item["wave_counts"], "wave_count_histogram"),
            (item["degree_histogram"], "degree_histogram"),
            (item["document_length_histogram"], "document_length_histogram"),
        ):
            entries = ({value: 1 for value in source} if isinstance(source, list) else source)
            if isinstance(source, list):
                entries = {}
                for value in source:
                    entries[value] = entries.get(value, 0) + 1
            for name, count in entries.items():
                normalized = str(name)
                aggregate[target][normalized] = aggregate[target].get(normalized, 0) + int(count)
    return aggregate


def _run(train_args: argparse.Namespace, tool_args: argparse.Namespace, case: CaseDefinition) -> None:
    synthetic, data_source, warmup, steps = _configure_benchmark_args(train_args, tool_args, case)
    train_cli._maybe_init_distributed(train_args)
    device = train_cli._select_device()
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    expected_world_size = train_args.dp_size * train_args.tp_size * train_args.pp_size * train_args.cp_size
    if world_size != expected_world_size:
        raise ValueError(
            f"torchrun world_size={world_size} does not match recipe mesh={expected_world_size} "
            f"(dp={train_args.dp_size}, pp={train_args.pp_size}, cp={train_args.cp_size}, tp={train_args.tp_size})"
        )
    mesh = MeshConfig(
        dp=train_args.dp_size,
        pp=train_args.pp_size,
        cp=train_args.cp_size,
        tp=train_args.tp_size,
        ep=train_args.ep_size,
    )
    dp_rank, _, _, _ = mesh.rank_coordinates(rank)
    torch.manual_seed(train_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_args.seed)
    core: RuntimeCore | None = None
    prefetch_metrics: _PrefetchMetrics | None = None
    try:
        if rank == 0:
            print(
                f"=== synthetic train-performance profile ===\n"
                f"source={case.name} recipe={case.recipe} runtime_spec={case.runtime_spec}\n"
                f"mesh=dp={mesh.dp} tp={mesh.tp} pp={mesh.pp} cp={mesh.cp} ep={mesh.ep} world_size={world_size}\n"
                f"synthetic={synthetic.layout} segments=[{synthetic.segment_count_min},{synthetic.segment_count_max}] "
                f"supervised=[{synthetic.supervised_fraction_min:.3f},{synthetic.supervised_fraction_max:.3f}] "
                f"data_source={data_source} delivery={synthetic.delivery} "
                f"warmup={warmup} steps={steps} lr={train_args.lr:g}",
                flush=True,
            )
        model = train_cli._build_model(train_args)
        core = train_cli._build_runtime(train_args, model, device)
        core.setup()
        core.model.train()
        train_args._benchmark_flops_per_token = core.state.static_metrics.get("perf/flops_per_token")
        data_paths: list[Path] | None = None
        data_format: str | None = None
        if data_source == "recipe":
            if not train_args.data:
                raise ValueError("--data-source recipe requires recipe --data (or pass it after --)")
            data_paths, batches, data_format = train_cli._build_dataloader(
                train_args,
                dp_rank=dp_rank,
                cp_token_planner=core.plan.cp_token_planner,
            )
            if train_args.data_prefetch_batches:
                batches = PrefetchDataLoader(batches)
            if rank == 0:
                print(
                    f"real data loader: format={data_format} paths={len(data_paths)} first={data_paths[0]}",
                    flush=True,
                )
        else:
            cp_batch_sharder = None
            if train_args.batch_data_cp_aware:
                if core.plan.cp_token_planner is None:
                    raise ValueError("--batch-data-cp-aware requires --cp-token-planner")
                cp_batch_sharder = ContextParallelBatchSharder(
                    planner=build_context_token_planner(core.plan.cp_token_planner),
                    world_size=train_args.cp_size,
                )
            batches = SyntheticBatchStream(
                batch_size=train_args.micro_batch_size,
                seq_len=train_args.seq_len,
                vocab_size=train_args.vocab_size,
                config=synthetic,
                # HDP is a collective schedule over DP ranks, so every HDP
                # participant must begin with the identical canonical batch.
                # Ordinary DP replicas instead receive distinct deterministic
                # samples; TP/PP/CP ranks share their DP replica's batch.
                seed=tool_args.seed if train_args.hdp_balanced else tool_args.seed + dp_rank,
                runtime_device=device,
                cp_batch_sharder=cp_batch_sharder,
            )
            if train_args.hdp_balanced:
                hdp_loader = ByteScaleHdpDataLoader(
                    batches,
                    hdp_rank=dp_rank,
                    hdp_world_size=train_args.dp_size,
                    config=train_cli._build_hdp_config(train_args),
                )
                prefetch_metrics = _PrefetchMetrics()
                batches = _TimedHdpSyntheticLoader(hdp_loader, prefetch_metrics)
            # Match the real-data ordering: generation -> HDP wave creation
            # (when enabled) -> one-batch CPU prefetch/pin.  Host delivery is
            # retained so this includes the realistic producer work.
            if train_args.data_prefetch_batches:
                batches = PrefetchDataLoader(batches)
        for step in range(1, warmup + 1):
            loss = _run_global_step(core, batches)
            _check_finite(loss, core, phase="warmup", step=step)
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        if dist.is_initialized():
            dist.barrier()

        local_seconds: list[float] = []
        losses: list[float] = []
        for step in range(1, steps + 1):
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            loss = _run_global_step(core, batches)
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            local_seconds.append(time.perf_counter() - started)
            losses.append(_check_finite(loss, core, phase="timed", step=step))

        critical_seconds = _timing_summary(local_seconds, device) if dist.is_initialized() else local_seconds
        peak_bytes = torch.tensor(
            [torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0],
            dtype=torch.float64,
            device=device,
        )
        if dist.is_initialized():
            dist.all_reduce(peak_bytes, op=dist.ReduceOp.MAX)
        # all_gather_object is collective: every rank participates, while only
        # rank 0 serializes the resulting aggregate.
        aggregated_prefetch_metrics = _aggregate_prefetch_metrics(prefetch_metrics)
        if rank == 0:
            _write_summary(
                output_dir=tool_args.output_dir,
                case=case,
                train_args=train_args,
                synthetic=synthetic,
                data_source=data_source,
                data_format=data_format,
                data_paths=data_paths,
                warmup=warmup,
                step_seconds=critical_seconds,
                peak_memory_bytes=float(peak_bytes.item()),
                world_size=world_size,
                final_loss=losses[-1],
                profiled=tool_args.profile,
                prefetch_metrics=aggregated_prefetch_metrics,
                synthetic_seed=tool_args.seed,
            )
    finally:
        close_batches = locals().get("batches")
        close_loader = getattr(close_batches, "close", None)
        if callable(close_loader):
            close_loader()
        if core is not None:
            core.close()
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


def _write_summary(
    *,
    output_dir: Path,
    case: CaseDefinition,
    train_args: argparse.Namespace,
    synthetic: SyntheticBatchConfig,
    data_source: str,
    data_format: str | None,
    data_paths: list[Path] | None,
    warmup: int,
    step_seconds: list[float],
    peak_memory_bytes: float,
    world_size: int,
    final_loss: float,
    profiled: bool,
    prefetch_metrics: dict[str, Any] | None,
    synthetic_seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    seconds = torch.tensor(step_seconds, dtype=torch.float64)
    dp_replicas = 1 if train_args.hdp_balanced else train_args.dp_size
    tokens_per_step = dp_replicas * train_args.micro_batch_size * train_args.seq_len * train_args.grad_accum_steps
    mean_seconds = float(seconds.mean())
    p50_seconds = float(torch.quantile(seconds, 0.50))
    p90_seconds = float(torch.quantile(seconds, 0.90))
    trace = _trace_summary(synthetic, train_args, synthetic_seed=synthetic_seed)
    actual_tokens_per_step = trace["active_tokens_per_step"] if trace is not None else tokens_per_step
    throughput = tokens_per_step / mean_seconds
    actual_throughput = actual_tokens_per_step / mean_seconds
    flops_per_token = None
    # RuntimeCore populated this through the same model interface used by train/cli.
    # It is not serialized in args, so reconstructing it here would be wasteful;
    # the caller writes it onto the namespace before invoking this helper.
    flops_per_token = getattr(train_args, "_benchmark_flops_per_token", None)
    tflops_per_gpu = None if flops_per_token is None else throughput * flops_per_token / world_size / 1e12
    payload = {
        "case": case.name,
        "description": case.description,
        "recipe": case.recipe,
        "runtime_spec_source": case.runtime_spec,
        # Store the exact model/parallel portion of the resolved configuration
        # beside every result, so a later paper figure is reproducible even if
        # the original YAML recipe changes.
        "resolved_runtime_spec": build_runtime_spec(train_args),
        "synthetic": {
            "layout": synthetic.layout,
            "segment_count_min": synthetic.segment_count_min if synthetic.layout == "packed" else None,
            "segment_count_max": synthetic.segment_count_max if synthetic.layout == "packed" else None,
            "supervised_fraction_min": synthetic.supervised_fraction_min if synthetic.layout == "packed" else None,
            "supervised_fraction_max": synthetic.supervised_fraction_max if synthetic.layout == "packed" else None,
            "max_padding_fraction": synthetic.max_padding_fraction if synthetic.layout == "packed" else None,
            "delivery": synthetic.delivery,
            "seed": synthetic_seed,
            "document_length_trace": None if synthetic.document_length_trace is None else [list(row) for row in synthetic.document_length_trace],
            "document_length_trace_sha256": None if synthetic.document_length_trace is None else _trace_hash(synthetic.document_length_trace),
        },
        "environment": _environment_metadata(),
        "data_source": data_source,
        "data_format": data_format,
        "data_paths": None if data_paths is None else [str(path) for path in data_paths],
        "warmup_steps": warmup,
        "timed_steps": len(step_seconds),
        "mesh": {
            "dp": train_args.dp_size,
            "tp": train_args.tp_size,
            "pp": train_args.pp_size,
            "cp": train_args.cp_size,
            "ep": train_args.ep_size,
            "world_size": world_size,
        },
        "batch": {
            "micro_batch_size": train_args.micro_batch_size,
            "seq_len": train_args.seq_len,
            "grad_accum_steps": train_args.grad_accum_steps,
            "global_tokens_per_step": tokens_per_step,
            "actual_active_tokens_per_step": actual_tokens_per_step,
            "non_padding_tokens_per_step": actual_tokens_per_step,
        },
        "optimizer": {"lr": train_args.lr, "weight_decay": train_args.weight_decay},
        "timing_seconds": {"mean": mean_seconds, "p50": p50_seconds, "p90": p90_seconds, "per_step": step_seconds},
        "throughput_tokens_per_second": throughput,
        "throughput_nominal_tokens_per_second": throughput,
        "throughput_actual_active_tokens_per_second": actual_throughput,
        "flops_per_token": flops_per_token,
        "tflops_per_gpu_dense_estimate": tflops_per_gpu,
        "trace_workload": trace,
        "hdp": {
            "enabled": train_args.hdp_balanced,
            "partition_tokens": train_args.hdp_partition_tokens,
            "cost_alpha": train_args.hdp_cost_alpha,
            "cost_beta": train_args.hdp_cost_beta,
            "cost_gamma": train_args.hdp_cost_gamma,
            "balance_delta": train_args.hdp_balance_delta,
            "loader_prefetch": (
                None if prefetch_metrics is None else {
                    "scope": "all_produced_batches_including_warmup_and_terminal_speculative_prefetch",
                    **prefetch_metrics,
                }
            ),
        },
        "peak_allocated_gib": peak_memory_bytes / 2**30,
        "final_loss": final_loss,
        "torch_profiler": profiled,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "─" * 72,
        f"Synthetic train-performance profile: {case.name}",
        f"Model/runtime : {train_args.model}; dp={train_args.dp_size} tp={train_args.tp_size} pp={train_args.pp_size} cp={train_args.cp_size} ep={train_args.ep_size}; ZeRO-{train_args.zero_stage}",
        f"Input source  : {data_source}" + (f" ({data_format}, {len(data_paths or [])} shard paths)" if data_source == "recipe" else ""),
        f"Synthetic     : {synthetic.layout}"
        + (
            f" (segments={synthetic.segment_count_min}..{synthetic.segment_count_max}; "
            f"supervised={synthetic.supervised_fraction_min:.2f}..{synthetic.supervised_fraction_max:.2f}; "
            f"max-pad={synthetic.max_padding_fraction:.2%})"
            if synthetic.layout == "packed"
            else " (dense pretraining labels)"
        )
        + f"; delivery={synthetic.delivery}",
        f"Global batch  : {train_args.micro_batch_size} micro-batch × {train_args.grad_accum_steps} accumulation"
        + (" (HDP global logical batch)" if train_args.hdp_balanced else f" × DP {train_args.dp_size}")
        + f"; nominal {tokens_per_step:,} tokens/step | active {actual_tokens_per_step:,}",
        f"Step time     : mean {mean_seconds:.3f}s | p50 {p50_seconds:.3f}s | p90 {p90_seconds:.3f}s",
        f"Throughput    : nominal {throughput:,.0f} tok/s | active {actual_throughput:,.0f} tok/s",
        f"Peak VRAM     : {peak_memory_bytes / 2**30:.2f} GiB (max rank)",
        f"Final loss    : {final_loss:.6g} (finite check only; lr={train_args.lr:g})",
    ]
    if tflops_per_gpu is not None:
        lines.append(f"Compute       : {tflops_per_gpu:.1f} TFLOP/s/GPU dense-estimate ({flops_per_token:.6g} FLOP/token estimator; not packed-attention FLOPs)")
    if profiled:
        lines.append(f"Trace         : {output_dir / 'traces'}")
    lines.append("─" * 72)
    text = "\n".join(lines) + "\n"
    print(text, flush=True)
    (output_dir / "summary.txt").write_text(text, encoding="utf-8")


def _trace_hash(trace: tuple[tuple[int, ...], ...]) -> str:
    canonical = json.dumps([list(row) for row in trace], separators=(",", ":"), sort_keys=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _trace_summary(
    synthetic: SyntheticBatchConfig,
    args: argparse.Namespace,
    *,
    synthetic_seed: int,
) -> dict[str, Any] | None:
    if synthetic.document_length_trace is None:
        return None
    lengths = [length for row in synthetic.document_length_trace for length in row]
    active_per_microbatch = sum(lengths)
    if active_per_microbatch > args.micro_batch_size * args.seq_len:
        raise ValueError("document_length_trace exceeds the logical batch token contract")
    length_histogram: dict[str, int] = {}
    degree_histogram: dict[str, int] = {}
    for length in lengths:
        length_histogram[str(length)] = length_histogram.get(str(length), 0) + 1
        if args.hdp_balanced:
            assert args.hdp_partition_tokens is not None
            degree = min(args.dp_size, math.ceil(length / args.hdp_partition_tokens))
            degree_histogram[str(degree)] = degree_histogram.get(str(degree), 0) + 1
    return {
        "seed": synthetic_seed,
        "trace_sha256": _trace_hash(synthetic.document_length_trace),
        "rows": [list(row) for row in synthetic.document_length_trace],
        "repeat_each_synthetic_microbatch": True,
        "document_count_per_microbatch": len(lengths),
        "document_length_histogram_per_microbatch": length_histogram,
        "hdp_degree_histogram_per_microbatch": degree_histogram if args.hdp_balanced else None,
        "active_tokens_per_microbatch": active_per_microbatch,
        "nominal_tokens_per_microbatch": args.micro_batch_size * args.seq_len,
        "active_tokens_per_step": active_per_microbatch * args.grad_accum_steps,
        "nominal_tokens_per_step": args.micro_batch_size * args.seq_len * args.grad_accum_steps,
    }


def _environment_metadata() -> dict[str, Any]:
    def _git(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    try:
        import flash_attn  # type: ignore[import-not-found]
        flash_version = getattr(flash_attn, "__version__", "unknown")
    except Exception:
        flash_version = None
    nccl = None
    if torch.cuda.is_available():
        try:
            nccl_version = torch.cuda.nccl.version()
            nccl = ".".join(str(part) for part in nccl_version) if isinstance(nccl_version, tuple) else str(nccl_version)
        except Exception:
            nccl = None
    return {
        "hostname": socket.gethostname(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "nccl_version": nccl,
        "flash_attn_version": flash_version,
        "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }


def main() -> None:
    tool_args = _parse_tool_args()
    cases = _load_cases(tool_args.case_file)
    if tool_args.list_cases:
        if not cases:
            print(f"No cases found in {tool_args.case_file}")
            return
        for name, case in cases.items():
            source = f"recipe={case.recipe}" if case.recipe is not None else f"runtime_spec={case.runtime_spec}"
            print(f"{name}: {case.description}\n  {source}; warmup={case.warmup}; steps={case.steps}; synthetic={case.synthetic_layout}")
        return
    case = _resolve_source(tool_args, cases)
    train_args = _load_training_args(case, tool_args.recipe_overrides)
    _run(train_args, tool_args, case)


if __name__ == "__main__":
    main()
