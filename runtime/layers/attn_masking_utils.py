from __future__ import annotations

import torch


def canonical_position_ids(
    position_ids: torch.Tensor | None,
    *,
    batch_size: int,
    seq_len: int,
    position_offset: int,
    device: torch.device,
) -> torch.Tensor:
    if position_ids is None:
        positions = torch.arange(position_offset, position_offset + seq_len, device=device, dtype=torch.long)
        return positions.unsqueeze(0).expand(batch_size, -1).contiguous()
    if position_ids.dim() == 1:
        if position_ids.numel() != seq_len:
            raise ValueError(f"position_ids length must match seq_len, got {position_ids.numel()} vs {seq_len}")
        return position_ids.to(device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1).contiguous()
    if position_ids.dim() == 2:
        if position_ids.size(0) != batch_size or position_ids.size(1) != seq_len:
            raise ValueError(
                "position_ids shape must match [batch, seq], "
                f"got {tuple(position_ids.shape)} vs ({batch_size}, {seq_len})"
            )
        return position_ids.to(device=device, dtype=torch.long).contiguous()
    raise ValueError(f"position_ids must have rank 1 or 2, got shape={tuple(position_ids.shape)}")


def canonical_sequence_ids(
    sequence_ids: torch.Tensor | None,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor | None:
    if sequence_ids is None:
        return None
    if sequence_ids.dim() == 1:
        if sequence_ids.numel() != seq_len:
            raise ValueError(f"sequence_ids length must match seq_len, got {sequence_ids.numel()} vs {seq_len}")
        return sequence_ids.to(device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1).contiguous()
    if sequence_ids.dim() == 2:
        if sequence_ids.size(0) != batch_size or sequence_ids.size(1) != seq_len:
            raise ValueError(
                "sequence_ids shape must match [batch, seq], "
                f"got {tuple(sequence_ids.shape)} vs ({batch_size}, {seq_len})"
            )
        return sequence_ids.to(device=device, dtype=torch.long).contiguous()
    raise ValueError(f"sequence_ids must have rank 1 or 2, got shape={tuple(sequence_ids.shape)}")


def build_example_causal_mask(
    *,
    q_positions: torch.Tensor,
    k_positions: torch.Tensor,
    q_sequence_ids: torch.Tensor | None = None,
    k_sequence_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if q_positions.dim() != 2 or k_positions.dim() != 2:
        raise ValueError(
            "q_positions and k_positions must be rank-2 [batch, seq] tensors, "
            f"got {tuple(q_positions.shape)} and {tuple(k_positions.shape)}"
        )
    if q_positions.size(0) != k_positions.size(0):
        raise ValueError(
            "q_positions and k_positions must have same batch size, "
            f"got {q_positions.size(0)} vs {k_positions.size(0)}"
        )
    mask = k_positions.unsqueeze(1) <= q_positions.unsqueeze(2)
    if q_sequence_ids is None and k_sequence_ids is None:
        return mask
    if q_sequence_ids is None or k_sequence_ids is None:
        raise ValueError("q_sequence_ids and k_sequence_ids must both be provided or both be None")
    if q_sequence_ids.shape != q_positions.shape:
        raise ValueError(
            "q_sequence_ids shape must match q_positions, "
            f"got {tuple(q_sequence_ids.shape)} vs {tuple(q_positions.shape)}"
        )
    if k_sequence_ids.shape != k_positions.shape:
        raise ValueError(
            "k_sequence_ids shape must match k_positions, "
            f"got {tuple(k_sequence_ids.shape)} vs {tuple(k_positions.shape)}"
        )
    same_sequence = q_sequence_ids.unsqueeze(2) == k_sequence_ids.unsqueeze(1)
    return mask & same_sequence
