from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

from state import iter_logical_checkpoint_tensors, load_logical_checkpoint, load_logical_tensor
from state.logical_checkpoint import LogicalCheckpointTensorReader


def test_hf_safetensors_index_loads_as_logical_checkpoint() -> None:
    with tempfile.TemporaryDirectory(prefix="hf_logical_") as tmp:
        checkpoint_dir = Path(tmp)
        tensors = {
            "model.embed_tokens.weight": torch.arange(6, dtype=torch.float32).view(2, 3),
            "model.layers.0.self_attn.q_proj.weight": torch.arange(9, dtype=torch.float32).view(3, 3),
            "lm_head.weight": torch.arange(12, dtype=torch.float32).view(3, 4),
        }
        save_file(tensors, str(checkpoint_dir / "model-00001-of-00001.safetensors"))
        index = {
            "metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
            "weight_map": {name: "model-00001-of-00001.safetensors" for name in tensors},
        }
        with (checkpoint_dir / "model.safetensors.index.json").open("w", encoding="utf-8") as f:
            json.dump(index, f)

        loaded = load_logical_checkpoint(checkpoint_dir)
        assert sorted(loaded) == [
            "embed_tokens.weight",
            "layers.0.self_attn.q_proj.weight",
            "lm_head.weight",
        ]
        assert torch.equal(loaded["embed_tokens.weight"], tensors["model.embed_tokens.weight"])
        assert torch.equal(
            loaded["layers.0.self_attn.q_proj.weight"],
            tensors["model.layers.0.self_attn.q_proj.weight"],
        )
        assert torch.equal(load_logical_tensor(checkpoint_dir, "embed_tokens.weight"), tensors["model.embed_tokens.weight"])
        assert dict(iter_logical_checkpoint_tensors(checkpoint_dir)).keys() == loaded.keys()
        reader = LogicalCheckpointTensorReader(checkpoint_dir)
        assert set(reader) == set(loaded)
        assert torch.equal(reader["embed_tokens.weight"], tensors["model.embed_tokens.weight"])


if __name__ == "__main__":
    test_hf_safetensors_index_loads_as_logical_checkpoint()
    print("PASS")
