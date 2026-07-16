from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class MatrixCase:
    name: str
    module_key: str
    args: dict[str, object]
    needs_checkpoint_dir: bool = False

    def to_namespace(self) -> argparse.Namespace:
        return argparse.Namespace(**self.args)

    def to_cli_args(self) -> list[str]:
        args: list[str] = []
        for key, value in self.args.items():
            if key == "checkpoint_dir" and value is None:
                continue
            if key == "reuse_tp_for_ep":
                if value is False:
                    args.append("--no-reuse-tp-for-ep")
                continue
            if key == "reuse_cp_for_ep":
                if value is False:
                    args.append("--no-reuse-cp-for-ep")
                continue
            if key == "disable_precision":
                if value is True:
                    args.append("--disable-precision")
                continue
            if key == "disable_grad_clip":
                if value is True:
                    args.append("--disable-grad-clip")
                continue
            if key == "packed_batch":
                if value is True:
                    args.append("--packed-batch")
                continue
            flag = f"--{key.replace('_', '-')}"
            args.extend([flag, str(value)])
        return args


MODULE_SCRIPTS = {
    "full_eq": "tests/tiny_transformer_full_stack_equivalence.py",
    "full_resume": "tests/tiny_transformer_full_stack_resume.py",
    "ep_full_eq": "tests/tiny_transformer_ep_full_stack_equivalence.py",
    "ep_full_resume": "tests/tiny_transformer_ep_full_stack_resume.py",
}

_PACKED_BATCH_CASE_NAMES = {
    "A/full_eq/z3",
    "B/full_resume/z3_afab_acc2",
    "F/ep_topology_tp_lt_ep_le_tp_cp/ep_full_eq_z3",
}


_BASE_ARGS: dict[str, dict[str, object]] = {
    "full_eq": {
        "world_size": 8,
        "dp_size": 2,
        "pp_size": 2,
        "cp_size": 1,
        "tp_size": 2,
        "pp_microbatches": 2,
        "pp_schedule": "afab",
        "zero_stage": 3,
        "cp_attn_core": "all_gather_kv",
        "master_addr": "127.0.0.1",
        "master_port": 29569,
        "backend": "gloo",
        "batch_size": 8,
        "seq_len": 32,
        "seed": 42,
    },
    "full_resume": {
        "world_size": 8,
        "dp_size": 2,
        "pp_size": 2,
        "cp_size": 1,
        "tp_size": 2,
        "pp_microbatches": 2,
        "pp_schedule": "afab",
        "grad_accum_steps": 1,
        "cp_attn_core": "all_gather_kv",
        "master_addr": "127.0.0.1",
        "master_port": 29630,
        "backend": "gloo",
        "global_batch_size": 8,
        "seq_len": 32,
        "seed": 42,
        "checkpoint_dir": None,
        "zero_stage": 3,
        "disable_precision": False,
        "disable_grad_clip": False,
    },
    "ep_full_eq": {
        "world_size": 8,
        "dp_size": 2,
        "pp_size": 2,
        "cp_size": 2,
        "tp_size": 1,
        "ep_size": 4,
        "pp_microbatches": 2,
        "pp_schedule": "afab",
        "zero_stage": 3,
        "cp_attn_core": "all_gather_kv",
        "reuse_tp_for_ep": True,
        "reuse_cp_for_ep": True,
        "master_addr": "127.0.0.1",
        "master_port": 29644,
        "backend": "gloo",
        "batch_size": 8,
        "seq_len": 32,
        "seed": 42,
    },
    "ep_full_resume": {
        "world_size": 8,
        "dp_size": 2,
        "pp_size": 2,
        "cp_size": 2,
        "tp_size": 1,
        "ep_size": 4,
        "pp_microbatches": 2,
        "pp_schedule": "afab",
        "grad_accum_steps": 1,
        "cp_attn_core": "all_gather_kv",
        "reuse_tp_for_ep": True,
        "reuse_cp_for_ep": True,
        "master_addr": "127.0.0.1",
        "master_port": 29635,
        "backend": "gloo",
        "global_batch_size": 8,
        "seq_len": 32,
        "seed": 42,
        "checkpoint_dir": None,
        "zero_stage": 3,
        "disable_precision": False,
        "disable_grad_clip": False,
    },
}


def _case(
    name: str,
    module_key: str,
    *,
    needs_checkpoint_dir: bool = False,
    **overrides: object,
) -> MatrixCase:
    args = dict(_BASE_ARGS[module_key])
    args.update(overrides)
    return MatrixCase(name=name, module_key=module_key, args=args, needs_checkpoint_dir=needs_checkpoint_dir)


def _with_packed_batch(case: MatrixCase) -> MatrixCase:
    if case.name not in _PACKED_BATCH_CASE_NAMES:
        return case
    args = dict(case.args)
    args["packed_batch"] = True
    return MatrixCase(
        name=case.name,
        module_key=case.module_key,
        args=args,
        needs_checkpoint_dir=case.needs_checkpoint_dir,
    )


def _with_cuda_attention_backend(case: MatrixCase) -> MatrixCase:
    if case.args.get("backend") != "nccl" or int(case.args.get("cp_size", 1)) <= 1:
        return case
    args = dict(case.args)
    args.setdefault("attention_backend", "auto")
    return MatrixCase(
        name=case.name,
        module_key=case.module_key,
        args=args,
        needs_checkpoint_dir=case.needs_checkpoint_dir,
    )


def build_full_stack_matrix_cases(
    *,
    backend: str,
    world_size: int,
    case_names: list[str] | None = None,
    blacklist_names: list[str] | None = None,
    case_filter: str | None = None,
    max_cases: int | None = None,
) -> list[MatrixCase]:
    cases: list[MatrixCase] = []

    def add(case: MatrixCase) -> None:
        cases.append(case)

    def add_full_eq(prefix: str, *, zero_stages: tuple[int, ...], onef1b_zero_stages: tuple[int, ...] = (), **common: object) -> None:
        for z in zero_stages:
            add(_case(f"{prefix}/full_eq/z{z}", "full_eq", backend=backend, world_size=world_size, zero_stage=z, **common))
        for z in onef1b_zero_stages:
            add(
                _case(
                    f"{prefix}/full_eq/1f1b_z{z}",
                    "full_eq",
                    backend=backend,
                    world_size=world_size,
                    pp_schedule="1f1b",
                    zero_stage=z,
                    **common,
                )
            )

    def add_full_resume_suite(
        prefix: str,
        *,
        zero_stages: tuple[int, ...],
        schedule_accums: tuple[tuple[str, int], ...],
        **common: object,
    ) -> None:
        for z in zero_stages:
            for schedule, accum in schedule_accums:
                suffix = f"z{z}_{schedule}_acc{accum}"
                overrides = dict(common)
                overrides.update(
                    backend=backend,
                    world_size=world_size,
                    zero_stage=z,
                    pp_schedule=schedule,
                    grad_accum_steps=accum,
                )
                add(_case(f"{prefix}/full_resume/{suffix}", "full_resume", needs_checkpoint_dir=True, **overrides))

    def add_ep_full_eq(prefix: str, *, zero_stages: tuple[int, ...], onef1b_zero_stages: tuple[int, ...], **common: object) -> None:
        for z in zero_stages:
            add(_case(f"{prefix}/ep_full_eq/z{z}", "ep_full_eq", backend=backend, world_size=world_size, zero_stage=z, **common))
        for z in onef1b_zero_stages:
            add(
                _case(
                    f"{prefix}/ep_full_eq/1f1b_z{z}",
                    "ep_full_eq",
                    backend=backend,
                    world_size=world_size,
                    pp_schedule="1f1b",
                    zero_stage=z,
                    **common,
                )
            )

    def add_ep_full_resume_suite(
        prefix: str,
        *,
        zero_stages: tuple[int, ...],
        schedule_accums: tuple[tuple[str, int], ...],
        **common: object,
    ) -> None:
        for z in zero_stages:
            for schedule, accum in schedule_accums:
                suffix = f"z{z}_{schedule}_acc{accum}"
                overrides = dict(common)
                overrides.update(
                    backend=backend,
                    world_size=world_size,
                    zero_stage=z,
                    pp_schedule=schedule,
                    grad_accum_steps=accum,
                )
                add(_case(f"{prefix}/ep_full_resume/{suffix}", "ep_full_resume", needs_checkpoint_dir=True, **overrides))

    # A
    add_full_eq(
        "A",
        zero_stages=(0, 1, 2, 3),
        onef1b_zero_stages=(0, 1, 2, 3),
        dp_size=2,
        pp_size=2,
        cp_size=2,
        tp_size=1,
        cp_attn_core="ring",
    )
    add_full_resume_suite(
        "A",
        zero_stages=(0, 1, 2, 3),
        schedule_accums=(("afab", 1), ("1f1b", 1), ("afab", 2), ("1f1b", 2)),
        dp_size=2, pp_size=2, cp_size=2, tp_size=1, cp_attn_core="ring",
    )
    add_ep_full_eq("A", zero_stages=(0, 1, 2, 3), onef1b_zero_stages=(0, 1, 2, 3), dp_size=2, pp_size=2, cp_size=2, tp_size=1, ep_size=4, cp_attn_core="ring")
    add_ep_full_resume_suite(
        "A",
        zero_stages=(0, 1, 2, 3),
        schedule_accums=(("afab", 1), ("1f1b", 1), ("afab", 2), ("1f1b", 2)),
        dp_size=2, pp_size=2, cp_size=2, tp_size=1, ep_size=4, cp_attn_core="ring",
    )

    # B
    add_full_eq(
        "B",
        zero_stages=(0, 1, 2, 3),
        onef1b_zero_stages=(0, 1, 2, 3),
        dp_size=2,
        pp_size=2,
        cp_size=1,
        tp_size=2,
    )
    add_full_resume_suite(
        "B",
        zero_stages=(0, 1, 2, 3),
        schedule_accums=(("afab", 1), ("1f1b", 1), ("afab", 2), ("1f1b", 2)),
        dp_size=2, pp_size=2, cp_size=1, tp_size=2,
    )
    add_ep_full_eq("B", zero_stages=(0, 1, 2, 3), onef1b_zero_stages=(0, 1, 2, 3), dp_size=2, pp_size=2, cp_size=1, tp_size=2, ep_size=4)
    add_ep_full_resume_suite(
        "B",
        zero_stages=(0, 1, 2, 3),
        schedule_accums=(("afab", 1), ("1f1b", 1), ("afab", 2), ("1f1b", 2)),
        dp_size=2, pp_size=2, cp_size=1, tp_size=2, ep_size=4,
    )

    # C
    add_full_eq(
        "C",
        zero_stages=(0, 1, 2, 3),
        dp_size=2,
        pp_size=1,
        cp_size=2,
        tp_size=2,
        pp_microbatches=1,
        cp_attn_core="ring",
    )
    add_full_resume_suite(
        "C",
        zero_stages=(0, 1, 2, 3),
        schedule_accums=(("afab", 1), ("afab", 2)),
        dp_size=2, pp_size=1, cp_size=2, tp_size=2, pp_microbatches=1, cp_attn_core="ring",
    )
    add_ep_full_eq(
        "C",
        zero_stages=(0, 1, 2, 3),
        onef1b_zero_stages=(),
        dp_size=2,
        pp_size=1,
        cp_size=2,
        tp_size=2,
        ep_size=8,
        pp_microbatches=1,
        cp_attn_core="ring",
    )
    add_ep_full_resume_suite(
        "C",
        zero_stages=(0, 1, 2, 3),
        schedule_accums=(("afab", 1), ("afab", 2)),
        dp_size=2, pp_size=1, cp_size=2, tp_size=2, ep_size=8, pp_microbatches=1, cp_attn_core="ring",
    )

    # D
    add(_case("D/full_eq/z0", "full_eq", backend=backend, world_size=world_size, dp_size=1, pp_size=2, cp_size=2, tp_size=2, zero_stage=0, cp_attn_core="ring"))
    add(_case("D/full_eq/1f1b_z0", "full_eq", backend=backend, world_size=world_size, dp_size=1, pp_size=2, cp_size=2, tp_size=2, pp_schedule="1f1b", zero_stage=0, cp_attn_core="ring"))
    for schedule, accum, global_batch_size in (("afab", 1, 8), ("1f1b", 1, 8), ("afab", 2, 4), ("1f1b", 2, 4)):
        add(
            _case(
                f"D/full_resume/z0_{schedule}_acc{accum}",
                "full_resume",
                needs_checkpoint_dir=True,
                backend=backend,
                world_size=world_size,
                dp_size=1,
                pp_size=2,
                cp_size=2,
                tp_size=2,
                zero_stage=0,
                pp_schedule=schedule,
                grad_accum_steps=accum,
                global_batch_size=global_batch_size,
                cp_attn_core="ring",
            )
        )
    add(_case("D/ep_full_eq/z0", "ep_full_eq", backend=backend, world_size=world_size, dp_size=1, pp_size=2, cp_size=2, tp_size=2, ep_size=4, zero_stage=0, cp_attn_core="ring"))
    add(_case("D/ep_full_eq/1f1b_z0", "ep_full_eq", backend=backend, world_size=world_size, dp_size=1, pp_size=2, cp_size=2, tp_size=2, ep_size=4, pp_schedule="1f1b", zero_stage=0, cp_attn_core="ring"))
    for schedule, accum, global_batch_size in (("afab", 1, 8), ("1f1b", 1, 8), ("afab", 2, 4), ("1f1b", 2, 4)):
        add(
            _case(
                f"D/ep_full_resume/z0_{schedule}_acc{accum}",
                "ep_full_resume",
                needs_checkpoint_dir=True,
                backend=backend,
                world_size=world_size,
                dp_size=1,
                pp_size=2,
                cp_size=2,
                tp_size=2,
                ep_size=4,
                zero_stage=0,
                pp_schedule=schedule,
                grad_accum_steps=accum,
                global_batch_size=global_batch_size,
                cp_attn_core="ring",
            )
        )

    # E/F/G: targeted EP-span coverage for the three reuse_tp/reuse_cp regimes
    # under the default reuse_tp_for_ep=True, reuse_cp_for_ep=True plan.
    #
    # E: EP <= TP
    add(_case("E/ep_topology_ep_le_tp/ep_full_eq_z3", "ep_full_eq", backend=backend, world_size=world_size, dp_size=2, pp_size=1, cp_size=1, tp_size=4, ep_size=2, pp_microbatches=1, zero_stage=3))
    add(
        _case(
            "E/ep_topology_ep_le_tp/ep_full_resume_z3_afab_acc2",
            "ep_full_resume",
            needs_checkpoint_dir=True,
            backend=backend,
            world_size=world_size,
            dp_size=2,
            pp_size=1,
            cp_size=1,
            tp_size=4,
            ep_size=2,
            pp_microbatches=1,
            zero_stage=3,
            pp_schedule="afab",
            grad_accum_steps=2,
        )
    )

    # F: TP < EP <= TP*CP
    add(_case("F/ep_topology_tp_lt_ep_le_tp_cp/ep_full_eq_z3", "ep_full_eq", backend=backend, world_size=world_size, dp_size=2, pp_size=1, cp_size=2, tp_size=2, ep_size=4, pp_microbatches=1, zero_stage=3, cp_attn_core="ring"))
    add(
        _case(
            "F/ep_topology_tp_lt_ep_le_tp_cp/ep_full_resume_z3_afab_acc2",
            "ep_full_resume",
            needs_checkpoint_dir=True,
            backend=backend,
            world_size=world_size,
            dp_size=2,
            pp_size=1,
            cp_size=2,
            tp_size=2,
            ep_size=4,
            pp_microbatches=1,
            zero_stage=3,
            pp_schedule="afab",
            grad_accum_steps=2,
            cp_attn_core="ring",
        )
    )

    # G: TP*CP < EP <= TP*CP*DP
    add(_case("G/ep_topology_tp_cp_lt_ep_le_tp_cp_dp/ep_full_eq_z3", "ep_full_eq", backend=backend, world_size=world_size, dp_size=2, pp_size=1, cp_size=2, tp_size=2, ep_size=8, pp_microbatches=1, zero_stage=3, cp_attn_core="ring"))
    add(
        _case(
            "G/ep_topology_tp_cp_lt_ep_le_tp_cp_dp/ep_full_resume_z3_afab_acc2",
            "ep_full_resume",
            needs_checkpoint_dir=True,
            backend=backend,
            world_size=world_size,
            dp_size=2,
            pp_size=1,
            cp_size=2,
            tp_size=2,
            ep_size=8,
            pp_microbatches=1,
            zero_stage=3,
            pp_schedule="afab",
            grad_accum_steps=2,
            cp_attn_core="ring",
        )
    )

    # H: explicit no-reuse coverage. These replace the old standalone
    # EP+CP/EP+PP targeted runners while keeping the matrix as the one
    # distributed transformer entry point.
    add(
        _case(
            "H/ep_no_reuse_cp/ep_full_eq_z1",
            "ep_full_eq",
            backend=backend,
            world_size=world_size,
            dp_size=2,
            pp_size=2,
            cp_size=2,
            tp_size=1,
            ep_size=2,
            zero_stage=1,
            cp_attn_core="ring",
            reuse_cp_for_ep=False,
        )
    )
    add(
        _case(
            "H/ep_no_reuse_tp/ep_full_eq_z1",
            "ep_full_eq",
            backend=backend,
            world_size=world_size,
            dp_size=2,
            pp_size=2,
            cp_size=1,
            tp_size=2,
            ep_size=2,
            zero_stage=1,
            reuse_tp_for_ep=False,
        )
    )
    add(
        _case(
            "H/ep_no_reuse_tp_cp/ep_full_eq_z1",
            "ep_full_eq",
            backend=backend,
            world_size=world_size,
            dp_size=2,
            pp_size=1,
            cp_size=2,
            tp_size=2,
            ep_size=2,
            pp_microbatches=1,
            zero_stage=1,
            cp_attn_core="ring",
            reuse_tp_for_ep=False,
            reuse_cp_for_ep=False,
        )
    )

    if case_names:
        ordered_names: list[str] = []
        seen: set[str] = set()
        for value in case_names:
            for name in value.split(","):
                normalized = name.strip()
                if not normalized or normalized in seen:
                    continue
                ordered_names.append(normalized)
                seen.add(normalized)
        case_by_name = {case.name: case for case in cases}
        missing = [name for name in ordered_names if name not in case_by_name]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(f"unknown matrix case name(s): {missing_str}")
        cases = [case_by_name[name] for name in ordered_names]
    if blacklist_names:
        excluded: set[str] = set()
        for value in blacklist_names:
            for name in value.split(","):
                normalized = name.strip()
                if normalized:
                    excluded.add(normalized)
        if excluded:
            cases = [case for case in cases if case.name not in excluded]
    if case_filter:
        cases = [case for case in cases if case_filter in case.name]
    if max_cases is not None:
        cases = cases[:max_cases]
    if not cases:
        raise ValueError("no matrix cases selected")
    return [_with_packed_batch(_with_cuda_attention_backend(case)) for case in cases]
