"""Context compaction utilities for fast inference on tiny devices.

This module keeps the useful part of a prompt or cache and drops the rest.
The goal is not magical accuracy. The goal is a shorter active context so
small devices spend less time attending to junk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass
class ContextCompressionConfig:
    """Policy for keeping a short active context."""

    max_tokens: int = 256
    keep_first: int = 8
    keep_recent: int = 64
    keep_every: int = 16
    min_unique_weight: float = 0.25

    def clamp(self) -> "ContextCompressionConfig":
        max_tokens = max(1, self.max_tokens)
        keep_first = max(0, min(self.keep_first, max_tokens))
        keep_recent = max(0, min(self.keep_recent, max_tokens))
        keep_every = max(1, self.keep_every)
        return ContextCompressionConfig(
            max_tokens=max_tokens,
            keep_first=keep_first,
            keep_recent=keep_recent,
            keep_every=keep_every,
            min_unique_weight=max(0.0, self.min_unique_weight),
        )


@dataclass
class CompressedContext:
    """Result of context compaction."""

    original_tokens: List[int]
    compressed_tokens: List[int]
    kept_indices: List[int]
    dropped_indices: List[int]
    compression_ratio: float

    @property
    def original_length(self) -> int:
        return len(self.original_tokens)

    @property
    def compressed_length(self) -> int:
        return len(self.compressed_tokens)


class ContextCompressor:
    """Keep the parts of a context that matter most.

    The heuristic is intentionally simple and deterministic:
    - always keep the first few tokens
    - always keep the most recent tokens
    - keep every Nth token through the middle
    - fill any remaining budget with tokens that look relatively rare
    """

    def __init__(self, config: Optional[ContextCompressionConfig] = None):
        self.config = (config or ContextCompressionConfig()).clamp()

    def compress_tokens(
        self,
        tokens: Sequence[int],
        importance_scores: Optional[Sequence[float]] = None,
    ) -> CompressedContext:
        original_tokens = list(tokens)
        if not original_tokens:
            return CompressedContext([], [], [], [], 1.0)

        if len(original_tokens) <= self.config.max_tokens:
            kept = list(range(len(original_tokens)))
            return CompressedContext(
                original_tokens=original_tokens,
                compressed_tokens=original_tokens.copy(),
                kept_indices=kept,
                dropped_indices=[],
                compression_ratio=1.0,
            )

        scores = self._build_scores(original_tokens, importance_scores)
        keep_indices = self._select_indices(original_tokens, scores)
        keep_indices = sorted(set(keep_indices))
        compressed_tokens = [original_tokens[i] for i in keep_indices]
        dropped = [i for i in range(len(original_tokens)) if i not in set(keep_indices)]
        ratio = len(original_tokens) / max(1, len(compressed_tokens))

        return CompressedContext(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            kept_indices=keep_indices,
            dropped_indices=dropped,
            compression_ratio=ratio,
        )

    def _build_scores(
        self,
        tokens: Sequence[int],
        importance_scores: Optional[Sequence[float]],
    ) -> List[float]:
        if importance_scores is not None and len(importance_scores) == len(tokens):
            return [float(v) for v in importance_scores]

        counts = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1

        scores: List[float] = []
        total = len(tokens)
        for idx, token in enumerate(tokens):
            recency = (idx + 1) / total
            rarity = 1.0 / counts[token]
            structural = 1.0 if idx < self.config.keep_first else 0.0
            scores.append(recency * 0.15 + rarity + structural)
        return scores

    def _select_indices(self, tokens: Sequence[int], scores: Sequence[float]) -> List[int]:
        n = len(tokens)
        cfg = self.config
        budget = min(cfg.max_tokens, n)

        keep = set(range(min(cfg.keep_first, n)))
        recent_start = max(0, n - cfg.keep_recent)
        keep.update(range(recent_start, n))

        middle_start = len(keep) and min(max(cfg.keep_first, 0), n)
        middle_end = recent_start
        for idx in range(middle_start, middle_end, cfg.keep_every):
            keep.add(idx)

        if len(keep) < budget:
            middle_candidates = [
                idx
                for idx in range(n)
                if idx not in keep
            ]
            middle_candidates.sort(key=lambda i: scores[i], reverse=True)
            for idx in middle_candidates:
                keep.add(idx)
                if len(keep) >= budget:
                    break

        if len(keep) > budget:
            ordered = sorted(keep, key=lambda i: (i < cfg.keep_first, scores[i], -i), reverse=True)
            keep = set(ordered[:budget])

        return sorted(keep)

    def compress_with_report(
        self,
        tokens: Sequence[int],
        importance_scores: Optional[Sequence[float]] = None,
    ) -> Tuple[List[int], CompressedContext]:
        report = self.compress_tokens(tokens, importance_scores)
        return report.compressed_tokens, report
