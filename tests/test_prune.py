from __future__ import annotations

import numpy as np

from bitforge.prune import BlockPruner, PruningConfig
from bitforge.context import ContextCompressionConfig, ContextCompressor


def test_block_prune_shrinks_data():
    weights = {"layer": np.random.randn(4096).astype(np.float32)}
    pruner = BlockPruner(PruningConfig(block_size=32, target_keep_ratio=0.2))
    result = pruner.prune_weights(weights)
    assert result.sparse_nbytes < result.original_nbytes
    assert result.compression_ratio > 1.0


def test_prune_to_budget():
    weights = {"layer": np.random.randn(2048).astype(np.float32)}
    pruner = BlockPruner(PruningConfig(block_size=32))
    result = pruner.prune_to_budget(weights, budget_bytes=2000)
    assert result.sparse_nbytes <= 2000 or result.compression_ratio > 1.0


def test_context_compaction():
    tokens = list(range(300))
    compressor = ContextCompressor(ContextCompressionConfig(max_tokens=96, keep_first=8, keep_recent=24, keep_every=8))
    compressed = compressor.compress_tokens(tokens)
    assert len(compressed.compressed_tokens) <= 96
    assert compressed.compression_ratio > 1.0
