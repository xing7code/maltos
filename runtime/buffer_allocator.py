from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch


_POOL_ALIGNMENT_BYTES = 256
_POOL_DEFAULT_SLAB_BYTES = 4 * 1024 * 1024


class BufferPolicy(str, Enum):
    PINNED = "pinned"
    CACHEABLE = "cacheable"
    TEMPORARY = "temporary"


@dataclass
class BufferHandle:
    tensor: torch.Tensor
    policy: BufferPolicy
    pool_key: tuple[BufferPolicy, torch.device, torch.dtype] | None = None
    slab_index: int | None = None
    start: int = 0
    alloc_numel: int = 0
    released: bool = False


@dataclass(order=True, frozen=True)
class _FreeSegment:
    start: int
    length: int


@dataclass
class _BufferSlab:
    storage: torch.Tensor
    free_segments: list[_FreeSegment]


@dataclass
class BufferPool:
    _pinned_handles: dict[tuple[str, torch.device, tuple[int, ...], torch.dtype], BufferHandle] = field(default_factory=dict)
    _slabs: dict[tuple[BufferPolicy, torch.device, torch.dtype], list[_BufferSlab]] = field(default_factory=dict)

    def acquire(
        self,
        *,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        policy: BufferPolicy | str,
        key: str | None = None,
    ) -> BufferHandle:
        resolved_policy = BufferPolicy(policy)
        resolved_device = torch.device(device)
        if resolved_policy == BufferPolicy.PINNED:
            if key is None:
                raise ValueError("pinned buffers require a key")
            return self._acquire_pinned(
                key=key,
                shape=shape,
                dtype=dtype,
                device=resolved_device,
            )
        if key is not None:
            raise ValueError(f"{resolved_policy.value} buffers do not accept a key")
        return self._acquire_slabbed(
            shape=shape,
            dtype=dtype,
            device=resolved_device,
            policy=resolved_policy,
        )

    def release(self, handle: BufferHandle) -> None:
        if handle.policy == BufferPolicy.PINNED:
            return
        if handle.released:
            raise ValueError("buffer handle has already been released")
        if handle.pool_key is None or handle.slab_index is None:
            raise ValueError("buffer handle is missing slab metadata")
        slabs = self._slabs.get(handle.pool_key)
        if slabs is None or handle.slab_index >= len(slabs):
            raise ValueError("buffer handle refers to an unknown slab")
        slab = slabs[handle.slab_index]
        slab.free_segments.append(_FreeSegment(start=handle.start, length=handle.alloc_numel))
        slab.free_segments = _coalesce_segments(slab.free_segments)
        handle.released = True

    def clear(self) -> None:
        self._pinned_handles.clear()
        self._slabs.clear()

    def _acquire_pinned(
        self,
        *,
        key: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> BufferHandle:
        handle_key = (key, device, shape, dtype)
        handle = self._pinned_handles.get(handle_key)
        if handle is not None:
            return handle
        requested_numel = _numel(shape)
        alloc_numel = _aligned_numel(requested_numel, dtype)
        storage = torch.empty((alloc_numel,), dtype=dtype, device=device)
        view = storage.narrow(0, 0, requested_numel).view(shape)
        handle = BufferHandle(
            tensor=view,
            policy=BufferPolicy.PINNED,
        )
        self._pinned_handles[handle_key] = handle
        return handle

    def _acquire_slabbed(
        self,
        *,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        policy: BufferPolicy,
    ) -> BufferHandle:
        pool_key = (policy, device, dtype)
        requested_numel = _numel(shape)
        alloc_numel = _aligned_numel(requested_numel, dtype)
        slabs = self._slabs.setdefault(pool_key, [])

        for slab_index, slab in enumerate(slabs):
            segment_index = _find_first_fit_segment(slab.free_segments, alloc_numel)
            if segment_index is None:
                continue
            start = _take_segment(slab.free_segments, segment_index, alloc_numel)
            view = slab.storage.narrow(0, start, requested_numel).view(shape)
            return BufferHandle(
                tensor=view,
                policy=policy,
                pool_key=pool_key,
                slab_index=slab_index,
                start=start,
                alloc_numel=alloc_numel,
            )

        slab_numel = max(alloc_numel, _default_slab_numel(dtype))
        storage = torch.empty((slab_numel,), dtype=dtype, device=device)
        slabs.append(
            _BufferSlab(
                storage=storage,
                free_segments=[_FreeSegment(start=alloc_numel, length=slab_numel - alloc_numel)]
                if alloc_numel < slab_numel
                else [],
            )
        )
        view = storage.narrow(0, 0, requested_numel).view(shape)
        return BufferHandle(
            tensor=view,
            policy=policy,
            pool_key=pool_key,
            slab_index=len(slabs) - 1,
            start=0,
            alloc_numel=alloc_numel,
        )


def _numel(shape: tuple[int, ...]) -> int:
    if not shape:
        return 1
    total = 1
    for dim in shape:
        total *= dim
    return total


def _aligned_numel(numel: int, dtype: torch.dtype) -> int:
    alignment = max(1, _POOL_ALIGNMENT_BYTES // torch.empty((), dtype=dtype).element_size())
    return ((numel + alignment - 1) // alignment) * alignment


def _default_slab_numel(dtype: torch.dtype) -> int:
    elem_size = torch.empty((), dtype=dtype).element_size()
    return max(1, _POOL_DEFAULT_SLAB_BYTES // elem_size)


def _find_first_fit_segment(segments: list[_FreeSegment], needed_numel: int) -> int | None:
    for index, segment in enumerate(segments):
        if segment.length >= needed_numel:
            return index
    return None


def _take_segment(segments: list[_FreeSegment], index: int, needed_numel: int) -> int:
    segment = segments[index]
    start = segment.start
    remaining = segment.length - needed_numel
    if remaining == 0:
        segments.pop(index)
        return start
    segments[index] = _FreeSegment(start=segment.start + needed_numel, length=remaining)
    return start


def _coalesce_segments(segments: list[_FreeSegment]) -> list[_FreeSegment]:
    if not segments:
        return []
    merged: list[_FreeSegment] = []
    for segment in sorted(segments):
        if not merged:
            merged.append(segment)
            continue
        prev = merged[-1]
        prev_end = prev.start + prev.length
        if prev_end == segment.start:
            merged[-1] = _FreeSegment(start=prev.start, length=prev.length + segment.length)
            continue
        merged.append(segment)
    return merged


_GLOBAL_BUFFER_POOL: BufferPool | None = None


def global_buffer_pool() -> BufferPool:
    global _GLOBAL_BUFFER_POOL
    if _GLOBAL_BUFFER_POOL is None:
        _GLOBAL_BUFFER_POOL = BufferPool()
    return _GLOBAL_BUFFER_POOL


def acquire_buffer(
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    policy: BufferPolicy | str,
    key: str | None = None,
) -> BufferHandle:
    return global_buffer_pool().acquire(
        shape=shape,
        dtype=dtype,
        device=device,
        policy=policy,
        key=key,
    )


def release_buffer(handle: BufferHandle) -> None:
    global_buffer_pool().release(handle)


def clear_buffer_pool() -> None:
    global _GLOBAL_BUFFER_POOL
    if _GLOBAL_BUFFER_POOL is not None:
        _GLOBAL_BUFFER_POOL.clear()
    _GLOBAL_BUFFER_POOL = None
