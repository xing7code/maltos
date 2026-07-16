from __future__ import annotations

import torch

from models.tiny_transformer import TinyTransformer
from parallel import ParallelPlan
from parallel.context_interfaces import ContextParallelAttentionCoreType
from runtime import MeshConfig, RuntimeCore
from runtime.plugins.cp import _shard_batch_for_cp
from utils.constants import IGNORE_INDEX, INPUT_IDS_KEY, LABELS_KEY, POSITION_IDS_KEY, SEQUENCE_IDS_KEY


def _sgd_factory():
    return lambda params: torch.optim.SGD(params, lr=1e-2)


def test_runtime_accepts_sft_batch_fields() -> None:
    model = TinyTransformer(
        dim=32,
        n_heads=4,
        n_kv_heads=4,
        hidden_size=64,
        eps=1e-5,
        n_layers=2,
        vocab_size=64,
        max_seq_len=16,
    )
    core = RuntimeCore(
        mesh=MeshConfig(),
        plan=ParallelPlan(),
        model=model,
        optimizer_factory=_sgd_factory(),
    )
    batch = {
        INPUT_IDS_KEY: torch.randint(0, 32, (2, 8)),
        LABELS_KEY: torch.randint(0, 32, (2, 8)),
        POSITION_IDS_KEY: torch.arange(8, dtype=torch.long).unsqueeze(0).expand(2, -1).contiguous(),
        SEQUENCE_IDS_KEY: torch.zeros((2, 8), dtype=torch.long),
    }
    batch[LABELS_KEY][:, -1] = IGNORE_INDEX

    core.setup()
    loss, _ = core.run_step(batch)
    core.step_optimizer()

    assert loss.ndim == 0
    assert core.state.step == 1


def test_cp_sharding_preserves_sequence_ids() -> None:
    batch = {
        INPUT_IDS_KEY: torch.tensor([[10, 11, 12, 13, 14, 15]], dtype=torch.long),
        LABELS_KEY: torch.tensor([[11, 12, 13, 14, 15, IGNORE_INDEX]], dtype=torch.long),
        POSITION_IDS_KEY: torch.tensor([[0, 1, 2, 0, 1, 2]], dtype=torch.long),
        SEQUENCE_IDS_KEY: torch.tensor([[7, 7, 7, 8, 8, 8]], dtype=torch.long),
    }

    sharded = _shard_batch_for_cp(
        batch,
        rank=1,
        world_size=2,
        attention_core_type=ContextParallelAttentionCoreType.ALL_GATHER_KV,
    )

    assert sharded[INPUT_IDS_KEY].tolist() == [[13, 14, 15]]
    assert sharded[LABELS_KEY].tolist() == [[14, 15, IGNORE_INDEX]]
    assert sharded[POSITION_IDS_KEY].tolist() == [[0, 1, 2]]
    assert sharded[SEQUENCE_IDS_KEY].tolist() == [[8, 8, 8]]


def main() -> None:
    test_runtime_accepts_sft_batch_fields()
    test_cp_sharding_preserves_sequence_ids()
    print("sft runtime fields ok")


if __name__ == "__main__":
    main()
