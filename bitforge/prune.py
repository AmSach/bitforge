"""Structured pruning for export to tiny devices.

This is the part that makes BitForge more useful on ESP/RPi-style targets:
- it removes low-value blocks first
- it can shrink a model export to a fixed byte budget
- it prefers recent, sensitive, and high-variance weights
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class PruningConfig:
    """Settings for block pruning."""

    block_size: int = 64
    target_keep_ratio: float = 0.25
    min_keep_ratio: float = 0.05
    max_keep_ratio: float = 1.0
    budget_bytes: Optional[int] = None
    preserve_first_blocks: int = 1
    preserve_last_blocks: int = 2

    def clamp(self) -> "PruningConfig":
        block_size = max(1, int(self.block_size))
        keep = min(max(self.target_keep_ratio, self.min_keep_ratio), self.max_keep_ratio)
        return PruningConfig(
            block_size=block_size,
            target_keep_ratio=keep,
            min_keep_ratio=max(0.0, self.min_keep_ratio),
            max_keep_ratio=max(keep, self.max_keep_ratio),
            budget_bytes=self.budget_bytes,
            preserve_first_blocks=max(0, int(self.preserve_first_blocks)),
            preserve_last_blocks=max(0, int(self.preserve_last_blocks)),
        )


@dataclass
class PrunedLayer:
    """A pruned layer export."""

    name: str
    original_shape: Tuple[int, ...]
    kept_blocks: int
    total_blocks: int
    keep_mask: np.ndarray
    compressed_data: np.ndarray
    original_nbytes: int
    sparse_nbytes: int

    @property
    def keep_ratio(self) -> float:
        return self.kept_blocks / max(1, self.total_blocks)


@dataclass
class PrunedModelResult:
    """Whole-model pruning result."""

    layers: Dict[str, PrunedLayer]
    original_nbytes: int
    sparse_nbytes: int
    compression_ratio: float
    keep_ratio: float


class BlockPruner:
    """Prune low-value blocks first, then pack the survivors."""

    def __init__(self, config: Optional[PruningConfig] = None):
        self.config = (config or PruningConfig()).clamp()

    def prune_weights(
        self,
        weights: Dict[str, np.ndarray],
    ) -> PrunedModelResult:
        layers: Dict[str, PrunedLayer] = {}
        original_nbytes = 0
        sparse_nbytes = 0

        for name, arr in weights.items():
            layer = self.prune_tensor(name, np.asarray(arr))
            layers[name] = layer
            original_nbytes += layer.original_nbytes
            sparse_nbytes += layer.sparse_nbytes

        compression_ratio = original_nbytes / max(1, sparse_nbytes)
        keep_ratio = sparse_nbytes / max(1, original_nbytes)
        return PrunedModelResult(
            layers=layers,
            original_nbytes=original_nbytes,
            sparse_nbytes=sparse_nbytes,
            compression_ratio=compression_ratio,
            keep_ratio=keep_ratio,
        )

    def prune_tensor(self, name: str, tensor: np.ndarray) -> PrunedLayer:
        flat = np.asarray(tensor, dtype=np.float32).reshape(-1)
        if flat.size == 0:
            raise ValueError(f"Cannot prune empty tensor: {name}")

        cfg = self.config
        block_size = cfg.block_size
        total_blocks = int(np.ceil(flat.size / block_size))
        pad = total_blocks * block_size - flat.size
        if pad:
            flat = np.pad(flat, (0, pad))

        blocks = flat.reshape(total_blocks, block_size)
        scores = self._score_blocks(blocks)

        keep_blocks = self._pick_blocks(scores)
        keep_mask = np.zeros(total_blocks, dtype=bool)
        keep_mask[keep_blocks] = True

        compressed = self._pack_sparse(blocks, keep_mask)
        original_nbytes = tensor.nbytes
        sparse_nbytes = compressed.nbytes + keep_mask.nbytes

        return PrunedLayer(
            name=name,
            original_shape=tuple(tensor.shape),
            kept_blocks=int(keep_mask.sum()),
            total_blocks=total_blocks,
            keep_mask=keep_mask,
            compressed_data=compressed,
            original_nbytes=original_nbytes,
            sparse_nbytes=sparse_nbytes,
        )

    def _score_blocks(self, blocks: np.ndarray) -> np.ndarray:
        mean = np.abs(blocks).mean(axis=1)
        var = blocks.var(axis=1)
        scores = mean + 0.5 * np.sqrt(var)
        return scores

    def _pick_blocks(self, scores: np.ndarray) -> List[int]:
        total_blocks = len(scores)
        cfg = self.config
        keep = set(range(min(cfg.preserve_first_blocks, total_blocks)))
        keep.update(range(max(0, total_blocks - cfg.preserve_last_blocks), total_blocks))

        desired = int(round(total_blocks * cfg.target_keep_ratio))
        desired = max(int(np.ceil(total_blocks * cfg.min_keep_ratio)), min(total_blocks, desired))

        order = np.argsort(scores)[::-1]
        for idx in order:
            keep.add(int(idx))
            if len(keep) >= desired:
                break

        return sorted(keep)

    def _pack_sparse(self, blocks: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
        kept = blocks[keep_mask].astype(np.float16, copy=False).reshape(-1)
        indices = np.flatnonzero(keep_mask).astype(np.int32).reshape(-1)
        header = np.array([blocks.shape[0], blocks.shape[1], len(indices)], dtype=np.int32).view(np.uint8).reshape(-1)
        index_bytes = indices.view(np.uint8).reshape(-1)
        value_bytes = kept.view(np.uint8).reshape(-1)
        return np.concatenate([header, index_bytes, value_bytes])

    def prune_to_budget(
        self,
        weights: Dict[str, np.ndarray],
        budget_bytes: int,
    ) -> PrunedModelResult:
        cfg = self.config.clamp()
        result = self.prune_weights(weights)
        if result.sparse_nbytes <= budget_bytes:
            return result

        for ratio in [0.2, 0.15, 0.1, 0.08, 0.05]:
            self.config = PruningConfig(
                block_size=cfg.block_size,
                target_keep_ratio=ratio,
                min_keep_ratio=cfg.min_keep_ratio,
                max_keep_ratio=cfg.max_keep_ratio,
                budget_bytes=budget_bytes,
                preserve_first_blocks=cfg.preserve_first_blocks,
                preserve_last_blocks=cfg.preserve_last_blocks,
            )
            result = self.prune_weights(weights)
            if result.sparse_nbytes <= budget_bytes:
                return result

        return result
