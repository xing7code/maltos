from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class _BufferSlot:
    tensor: torch.Tensor


@dataclass
class _BufferAllocator:
    _slots: dict[tuple[str, torch.device, tuple[int, ...], torch.dtype], _BufferSlot] = field(default_factory=dict)

    def get(
        self,
        *,
        key: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        slot_key = (key, device, shape, dtype)
        slot = self._slots.get(slot_key)
        if slot is None:
            tensor = torch.empty(shape, dtype=dtype, device=device)
            self._slots[slot_key] = _BufferSlot(tensor=tensor)
            return tensor
        return slot.tensor


_GLOBAL_BUFFER_ALLOCATOR: _BufferAllocator | None = None


def allocate_buffer(
    *,
    key: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    global _GLOBAL_BUFFER_ALLOCATOR
    if _GLOBAL_BUFFER_ALLOCATOR is None:
        _GLOBAL_BUFFER_ALLOCATOR = _BufferAllocator()
    return _GLOBAL_BUFFER_ALLOCATOR.get(
        key=key,
        shape=shape,
        dtype=dtype,
        device=device,
    )
