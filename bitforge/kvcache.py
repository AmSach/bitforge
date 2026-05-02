"""KV cache compression for small-device inference.

This is the piece that BitForge was missing to make the project feel like a
real ESP/Raspberry Pi stack rather than just a weight compressor.

The approach is deliberately practical:
- keep key/value tensors compressible
- preserve shapes and metadata
- support layer-by-layer packing and unpacking
- track memory savings and latency

It is not a magic speed button. The win comes from reducing cache size,
bandwidth, and memory churn on long generations.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .compress.quantize import Quantizer


@dataclass
class KVCacheConfig:
    """Settings for KV cache compression."""

    recent_bits: int = 4
    mid_bits: int = 3
    old_bits: int = 2
    windows: Tuple[int, int, int] = (256, 1024, 4096)
    target_dtype: str = "float16"
    enable_profiling: bool = True

    def clamp(self) -> "KVCacheConfig":
        bits = {2, 3, 4, 8}
        recent = self.recent_bits if self.recent_bits in bits else 4
        mid = self.mid_bits if self.mid_bits in bits else 3
        old = self.old_bits if self.old_bits in bits else 2
        windows = tuple(max(1, int(v)) for v in self.windows)
        return KVCacheConfig(
            recent_bits=recent,
            mid_bits=mid,
            old_bits=old,
            windows=windows,
            target_dtype=self.target_dtype,
            enable_profiling=self.enable_profiling,
        )


@dataclass
class PackedTensor:
    """Compressed representation of a tensor."""

    data: np.ndarray
    bits: int
    scale: float
    zero_point: float
    shape: Tuple[int, ...]
    dtype: str
    original_numel: int
    packed_bytes: int

    @property
    def nbytes(self) -> int:
        return int(self.data.nbytes)


@dataclass
class PackedKVLayer:
    """Compressed key/value pair for one layer."""

    layer_idx: int
    key: PackedTensor
    value: PackedTensor
    seq_len: int

    @property
    def packed_bytes(self) -> int:
        return self.key.nbytes + self.value.nbytes


@dataclass
class PackedPastKeyValues:
    """Whole-cache compressed result."""

    layers: List[PackedKVLayer]
    original_bytes: int
    packed_bytes: int
    compression_ratio: float
    avg_latency_ms: float
    tokens_processed: int
    config: KVCacheConfig

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layers": [
                {
                    "layer_idx": layer.layer_idx,
                    "seq_len": layer.seq_len,
                    "key": {
                        "bits": layer.key.bits,
                        "shape": layer.key.shape,
                        "dtype": layer.key.dtype,
                        "scale": layer.key.scale,
                        "zero_point": layer.key.zero_point,
                        "original_numel": layer.key.original_numel,
                        "packed_bytes": layer.key.packed_bytes,
                    },
                    "value": {
                        "bits": layer.value.bits,
                        "shape": layer.value.shape,
                        "dtype": layer.value.dtype,
                        "scale": layer.value.scale,
                        "zero_point": layer.value.zero_point,
                        "original_numel": layer.value.original_numel,
                        "packed_bytes": layer.value.packed_bytes,
                    },
                }
                for layer in self.layers
            ],
            "original_bytes": self.original_bytes,
            "packed_bytes": self.packed_bytes,
            "compression_ratio": self.compression_ratio,
            "avg_latency_ms": self.avg_latency_ms,
            "tokens_processed": self.tokens_processed,
            "config": asdict(self.config),
        }


class KVCacheCompressor:
    """Compress and restore key/value caches.

    The compressor keeps the API intentionally small:
    - compress one layer
    - compress a whole cache
    - restore a whole cache

    It can be used directly in a demo loop or wrapped around a model-level
    cache handoff.
    """

    def __init__(self, config: Optional[KVCacheConfig] = None):
        self.config = (config or KVCacheConfig()).clamp()
        self.quantizer = Quantizer()
        self._stats = {
            "original_bytes": 0,
            "packed_bytes": 0,
            "latency_ms": 0.0,
            "tokens_processed": 0,
        }

    def allocate_bits(self, seq_len: int) -> torch.Tensor:
        """Allocate bits across sequence positions.

        The implementation stays simple: recent tokens get more precision,
        distant tokens get less.
        """
        cfg = self.config
        bits = torch.full((seq_len,), cfg.old_bits, dtype=torch.int8)
        recent_end, mid_end, _ = cfg.windows
        recent_actual = min(seq_len, recent_end)
        bits[max(0, seq_len - recent_actual) :] = cfg.recent_bits
        if seq_len > recent_end:
            mid_start = max(0, seq_len - mid_end)
            mid_stop = seq_len - recent_actual
            if mid_start < mid_stop:
                bits[mid_start:mid_stop] = cfg.mid_bits
        return bits

    def choose_bits(self, seq_len: int) -> int:
        allocation = self.allocate_bits(seq_len)
        avg_bits = int(round(float(allocation.float().mean().item())))
        return avg_bits if avg_bits in {2, 3, 4, 8} else 4

    def compress_tensor(self, tensor: torch.Tensor, bits: Optional[int] = None) -> PackedTensor:
        seq_len = tensor.shape[2] if tensor.dim() >= 3 else tensor.shape[0]
        selected_bits = bits or self.choose_bits(seq_len)
        packed, scale, zero_point = self.quantizer.quantize_tensor(tensor, selected_bits)
        return PackedTensor(
            data=packed,
            bits=selected_bits,
            scale=float(scale),
            zero_point=float(zero_point),
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype).replace("torch.", ""),
            original_numel=int(tensor.numel()),
            packed_bytes=(int(tensor.numel()) * selected_bits + 7) // 8,
        )

    def decompress_tensor(self, packed: PackedTensor) -> torch.Tensor:
        arr = self.quantizer.dequantize_tensor(
            packed.data,
            packed.bits,
            packed.scale,
            packed.zero_point,
            packed.shape,
        )
        target_dtype = getattr(torch, packed.dtype, torch.float16)
        return torch.from_numpy(arr).to(target_dtype)

    def compress_layer(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor) -> PackedKVLayer:
        seq_len = key.shape[2] if key.dim() >= 3 else key.shape[0]
        key_packed = self.compress_tensor(key, self.choose_bits(seq_len))
        value_packed = self.compress_tensor(value, self.choose_bits(seq_len))
        original_bytes = key.numel() * key.element_size() + value.numel() * value.element_size()
        packed_bytes = key_packed.nbytes + value_packed.nbytes
        if self.config.enable_profiling:
            self._stats["original_bytes"] += original_bytes
            self._stats["packed_bytes"] += packed_bytes
            self._stats["tokens_processed"] += seq_len
        return PackedKVLayer(layer_idx=layer_idx, key=key_packed, value=value_packed, seq_len=seq_len)

    def decompress_layer(self, packed_layer: PackedKVLayer) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.decompress_tensor(packed_layer.key), self.decompress_tensor(packed_layer.value)

    def compress_past_key_values(
        self,
        past_key_values: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    ) -> PackedPastKeyValues:
        import time

        start = time.perf_counter()
        layers: List[PackedKVLayer] = []
        original_bytes = 0
        packed_bytes = 0
        tokens_processed = 0

        for idx, (key, value) in enumerate(past_key_values):
            layer = self.compress_layer(idx, key, value)
            layers.append(layer)
            original_bytes += key.numel() * key.element_size() + value.numel() * value.element_size()
            packed_bytes += layer.packed_bytes
            tokens_processed += layer.seq_len

        latency_ms = (time.perf_counter() - start) * 1000
        if self.config.enable_profiling:
            self._stats["latency_ms"] += latency_ms

        ratio = original_bytes / max(1, packed_bytes)
        avg_latency = latency_ms / max(1, len(layers))
        return PackedPastKeyValues(
            layers=layers,
            original_bytes=original_bytes,
            packed_bytes=packed_bytes,
            compression_ratio=ratio,
            avg_latency_ms=avg_latency,
            tokens_processed=tokens_processed,
            config=self.config,
        )

    def decompress_past_key_values(
        self,
        packed: PackedPastKeyValues,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return [self.decompress_layer(layer) for layer in packed.layers]

    def stats(self) -> Dict[str, float]:
        packed = max(1, self._stats["packed_bytes"])
        ratio = self._stats["original_bytes"] / packed
        return {
            "original_bytes": float(self._stats["original_bytes"]),
            "packed_bytes": float(self._stats["packed_bytes"]),
            "compression_ratio": float(ratio),
            "latency_ms": float(self._stats["latency_ms"]),
            "tokens_processed": float(self._stats["tokens_processed"]),
        }
