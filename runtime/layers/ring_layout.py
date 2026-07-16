from __future__ import annotations

import torch


def expected_zigzag_ring_positions(
    local_seq_len: int,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
) -> torch.Tensor:
    if local_seq_len % 2 != 0:
        raise ValueError(
            "zigzag ring attention requires an even local sequence length, "
            f"got local_seq_len={local_seq_len}"
        )
    half_len = local_seq_len // 2
    front_start = rank * half_len
    back_start = (2 * world_size - rank - 1) * half_len
    front = torch.arange(front_start, front_start + half_len, device=device, dtype=torch.long)
    back = torch.arange(back_start, back_start + half_len, device=device, dtype=torch.long)
    return torch.cat([front, back], dim=0)


def has_zigzag_ring_layout(
    position_ids: torch.Tensor,
    *,
    rank: int,
    world_size: int,
) -> bool:
    if position_ids.dim() != 2:
        return False
    try:
        expected = expected_zigzag_ring_positions(
            position_ids.size(1),
            rank=rank,
            world_size=world_size,
            device=position_ids.device,
        )
    except ValueError:
        return False
    return bool(torch.equal(position_ids, expected.unsqueeze(0).expand_as(position_ids)))
