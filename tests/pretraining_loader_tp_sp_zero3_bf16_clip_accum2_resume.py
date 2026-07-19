"""Resume test for pretraining loader + TP/SP + ZeRO3 + bf16 + grad clip + accumulation."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from data import PretrainingDataLoader, TokenShardDataset
from distributed_test_utils import (
    max_diff as _max_diff,
    named_tensors as _named_tensors,
    normalize_param_name as _normalize_param_name,
    rule_by_param_name as _rule_by_param_name,
    supports_bf16_autocast as _supports_bf16_autocast,
)
from models import TinyTransformer, TinyTransformerTpSp
from models.tiny_transformer import RmsNorm
from parallel import ParallelPlan
from runtime import MeshAxis, MeshConfig, RuntimeCore
from runtime.plugins.fp16 import Fp16Plugin
from runtime.plugins.sp import SequenceParallelPlugin
from runtime.plugins.tp import TensorParallelPlugin
from runtime.plugins.zero3 import Zero3Plugin
from state import load_sharded_checkpoint, save_sharded_checkpoint
from utils.constants import INPUT_IDS_KEY, LABELS_KEY


_MODEL_KWARGS = dict(
    dim=64,
    n_heads=4,
    n_kv_heads=4,
    hidden_size=128,
    eps=1e-5,
    n_layers=2,
    vocab_size=256,
    max_seq_len=64,
)
_LOSS_ATOL = 5e-2
_STEP_ATOL = 5e-2
_LR = 1e-2
_ZERO3_WRAP_CLS = {torch.nn.Linear, torch.nn.Embedding, RmsNorm}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--dp-size", type=int, default=2)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29540)
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    return parser.parse_args()


def _mesh_indices(rank: int, tp_size: int) -> tuple[int, int]:
    return rank // tp_size, rank % tp_size


def _build_reference(seed: int) -> TinyTransformer:
    torch.manual_seed(seed)
    return TinyTransformer(**_MODEL_KWARGS)


def _build_loader(seq_len: int, dp_idx: int, dp_size: int, seed: int) -> PretrainingDataLoader:
    testdata_dir = Path(__file__).parent / "testdata"
    dataset = TokenShardDataset(
        [
            testdata_dir / "tokens_00000.bin",
            testdata_dir / "tokens_00001.bin",
        ]
    )
    return PretrainingDataLoader(
        dataset,
        seq_len=seq_len,
        micro_batch_size=1,
        dp_rank=dp_idx,
        dp_world_size=dp_size,
        seed=seed,
    )


def _batch_tuple(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return batch[INPUT_IDS_KEY], batch[LABELS_KEY]

def _build_runtime(model: TinyTransformerTpSp, dp_size: int, tp_size: int) -> tuple[RuntimeCore, Zero3Plugin]:
    zero3 = Zero3Plugin(
        wrap_cls=_ZERO3_WRAP_CLS,
    )
    core = RuntimeCore(
        mesh=MeshConfig(dp=dp_size, tp=tp_size, pp=1, cp=1, ep=1),
        plan=ParallelPlan(),
        model=model,
        grad_accum_steps=2,
        grad_clip_max_norm=1.0,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=_LR),
        plugins=[
            TensorParallelPlugin(),
            SequenceParallelPlugin(),
            zero3,
        ],
        dtype=torch.bfloat16,
    )
    core.setup()
    return core, zero3


def _run_worker(rank: int, args: argparse.Namespace) -> None:
    dist.init_process_group(
        backend=args.backend,
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=rank,
        world_size=args.world_size,
    )
    if args.world_size != args.dp_size * args.tp_size:
        raise ValueError("world size must equal dp_size * tp_size")
    if not _supports_bf16_autocast():
        if rank == 0:
            print("SKIP: bf16 autocast is not supported on this runtime")
        dist.destroy_process_group()
        return

    dp_idx, _ = _mesh_indices(rank, args.tp_size)
    baseline_model = _build_reference(args.seed)

    continuous_model = TinyTransformerTpSp(**_MODEL_KWARGS)
    continuous_model.load_state_dict(baseline_model.state_dict())
    continuous_core, continuous_zero3 = _build_runtime(continuous_model, args.dp_size, args.tp_size)
    continuous_loader = _build_loader(args.seq_len, dp_idx, args.dp_size, args.seed)
    continuous_core.state_manager.bind_dataloader(continuous_loader)

    restored_model = TinyTransformerTpSp(**_MODEL_KWARGS)
    restored_model.load_state_dict(baseline_model.state_dict())
    restored_core, restored_zero3 = _build_runtime(restored_model, args.dp_size, args.tp_size)
    restored_loader = _build_loader(args.seq_len, dp_idx, args.dp_size, args.seed)
    restored_core.state_manager.bind_dataloader(restored_loader)

    first_batch = continuous_loader.next_batch()
    loss0_cont, _ = continuous_core.run_step(_batch_tuple(first_batch))
    loss0_cont = loss0_cont.detach()
    if continuous_core.state.step != 0 or continuous_core.state.step_context.microbatch_idx != 1:
        raise AssertionError("continuous core must be at mid-step state before checkpoint")
    saved_loader_state = continuous_loader.state_dict()
    save_sharded_checkpoint(
        continuous_core.state_manager,
        args.checkpoint_dir,
    )
    if dist.is_initialized():
        dist.barrier()

    loss_pairs = []
    for tag in ("A1", "B0", "B1"):
        batch = continuous_loader.next_batch()
        cont_loss, should_step = continuous_core.run_step(_batch_tuple(batch))
        cont_loss = cont_loss.detach()
        if should_step:
            continuous_core.step_optimizer()
        loss_pairs.append((tag, cont_loss, batch[INPUT_IDS_KEY].detach().clone()))

    load_sharded_checkpoint(restored_core.state_manager, args.checkpoint_dir)
    if restored_core.state.step != 0 or restored_core.state.step_context.microbatch_idx != 1:
        raise AssertionError("restored core must recover mid-step state")
    if restored_loader.state_dict() != saved_loader_state:
        raise AssertionError("restored dataloader did not recover saved cursor")

    restored_loss_pairs = []
    for tag in ("A1", "B0", "B1"):
        batch = restored_loader.next_batch()
        restored_loss, should_step = restored_core.run_step(_batch_tuple(batch))
        restored_loss = restored_loss.detach()
        if should_step:
            restored_core.step_optimizer()
        restored_loss_pairs.append((tag, restored_loss, batch[INPUT_IDS_KEY].detach().clone()))

    dp_group = continuous_core.get_group(MeshAxis.DP)
    reduced_loss_diffs: list[tuple[str, float]] = []
    batch_diffs: list[tuple[str, float]] = []
    for (tag, cont_loss, cont_batch), (_, restored_loss, restored_batch) in zip(loss_pairs, restored_loss_pairs):
        lhs_v = cont_loss.clone()
        rhs_v = restored_loss.clone()
        if dp_group is not None:
            dist.all_reduce(lhs_v, op=dist.ReduceOp.AVG, group=dp_group)
            dist.all_reduce(rhs_v, op=dist.ReduceOp.AVG, group=dp_group)
        reduced_loss_diffs.append((tag, abs(lhs_v.item() - rhs_v.item())))
        batch_diffs.append((tag, (cont_batch - restored_batch).abs().max().item()))

    continuous_zero3.materialize_model()
    restored_zero3.materialize_model()
    shard_rules = _rule_by_param_name(continuous_model)
    tp_group = continuous_core.get_group(MeshAxis.TP)
    continuous_params = _named_tensors(continuous_core.model, shard_rules, tp_group, normalize_name=_normalize_param_name)
    restored_params = _named_tensors(restored_core.model, shard_rules, tp_group, normalize_name=_normalize_param_name)
    param_name, param_diff = _max_diff(continuous_params, restored_params)

    if rank == 0:
        worst_loss_tag, worst_loss_diff = max(reduced_loss_diffs, key=lambda item: item[1])
        worst_batch_tag, worst_batch_diff = max(batch_diffs, key=lambda item: item[1])
        print(f"Checkpoint dir      : {args.checkpoint_dir}")
        print(f"Loader batch diff   : {worst_batch_diff:.2e}  ({worst_batch_tag})")
        print(f"Resume loss diff    : {worst_loss_diff:.2e}  ({worst_loss_tag}, atol={_LOSS_ATOL:.2e})")
        print(f"Resume param diff   : {param_diff:.2e}  ({param_name}, atol={_STEP_ATOL:.2e})")
        if worst_batch_diff > 0.0:
            raise AssertionError(f"pretraining dataloader resume mismatch: tag={worst_batch_tag}, diff={worst_batch_diff}")
        if worst_loss_diff > _LOSS_ATOL:
            raise AssertionError(f"loss mismatch after resume: tag={worst_loss_tag}, diff={worst_loss_diff:.2e}")
        if param_diff > _STEP_ATOL:
            raise AssertionError(f"param mismatch after resume: {param_name} diff={param_diff:.2e}")
        if loss0_cont.numel() != 1:
            raise AssertionError("first loss should be scalar")
        print("PASS")

    continuous_zero3.reshard_model()
    restored_zero3.reshard_model()
    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if args.checkpoint_dir is None:
        args.checkpoint_dir = tempfile.mkdtemp(prefix="pretrain_tp_sp_zero3_bf16_clip_accum2_resume_")
    mp.spawn(_run_worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
