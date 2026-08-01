from __future__ import annotations

from unittest import mock

import torch
import torch.distributed as dist
import torch.nn as nn

from runtime.layers.functional import (
    native_comm_overlap_workspace_bytes,
    prewarm_native_comm_overlap_workspace,
)
from runtime.plugins.tp_sp import TpSpPlugin
from runtime.types import SetupPhase
from train.cli import _build_runtime
from train.flags import parse_args_from


def _workspace_size() -> None:
    mb1 = native_comm_overlap_workspace_bytes(
        micro_batch_size=1,
        seq_len=4096,
        hidden_size=5120,
        tp_size=2,
        dtype=torch.bfloat16,
    )
    mb2 = native_comm_overlap_workspace_bytes(
        micro_batch_size=2,
        seq_len=4096,
        hidden_size=5120,
        tp_size=2,
        dtype=torch.bfloat16,
    )
    assert mb1 == 40 * 2**20
    assert mb2 == 80 * 2**20

    try:
        native_comm_overlap_workspace_bytes(
            micro_batch_size=1,
            seq_len=4095,
            hidden_size=5120,
            tp_size=2,
            dtype=torch.bfloat16,
        )
    except ValueError as exc:
        assert "must be divisible" in str(exc)
    else:
        raise AssertionError("non-divisible SP sequence length must fail")


def _prewarm_supported() -> None:
    from torch.distributed import _symmetric_memory as symm_mem

    group = object()
    workspace = mock.Mock()
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(torch.cuda, "synchronize") as synchronize,
        mock.patch.object(dist, "get_backend", return_value="nccl"),
        mock.patch.object(dist, "_get_process_group_name", return_value="tp_group", create=True),
        mock.patch.object(symm_mem, "get_symm_mem_workspace", return_value=workspace) as get_workspace,
    ):
        enabled = prewarm_native_comm_overlap_workspace(
            group,  # type: ignore[arg-type]
            min_workspace_bytes=80 * 2**20,
        )

    assert enabled is True
    get_workspace.assert_called_once_with("tp_group", 80 * 2**20)
    workspace.barrier.assert_called_once_with()
    synchronize.assert_called_once_with()


def _prewarm_fallback() -> None:
    group = object()
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(dist, "get_backend", return_value="gloo"),
        mock.patch.object(dist, "_get_process_group_name", create=True) as get_group_name,
    ):
        enabled = prewarm_native_comm_overlap_workspace(
            group,  # type: ignore[arg-type]
            min_workspace_bytes=1,
        )
    assert enabled is False
    get_group_name.assert_not_called()


def _train_cli_wiring() -> None:
    args = parse_args_from(
        [
            "--config",
            "configs/llama_10m.yaml",
            "--tp-native-comm-overlap",
        ],
        require_data=False,
    )
    runtime = _build_runtime(args, nn.Linear(1, 1), torch.device("cpu"))
    plugin = next(plugin for plugin in runtime.plugins if isinstance(plugin, TpSpPlugin))
    assert plugin.native_comm_overlap is True
    assert plugin.native_comm_overlap_workspace_bytes == 384 * 2**10


def _plugin_finalization_prewarm() -> None:
    group = object()
    runtime = mock.Mock()
    runtime.get_group.return_value = group
    plugin = TpSpPlugin(
        native_comm_overlap=True,
        native_comm_overlap_workspace_bytes=80 * 2**20,
    )
    plugin.bind(runtime)
    with mock.patch(
        "runtime.plugins.tp_sp.prewarm_native_comm_overlap_workspace",
        return_value=True,
    ) as prewarm:
        plugin.on_setup_phase(SetupPhase.FINALIZE, nn.Linear(1, 1))
    prewarm.assert_called_once_with(group, min_workspace_bytes=80 * 2**20)
    assert plugin.native_comm_overlap_workspace_prewarmed is True


def main() -> None:
    _workspace_size()
    _prewarm_supported()
    _prewarm_fallback()
    _train_cli_wiring()
    _plugin_finalization_prewarm()
    print("TP native-overlap workspace prewarm: PASS")


if __name__ == "__main__":
    main()
