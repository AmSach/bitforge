"""
BitForge - Shrink any LLM to fit in your pocket.

Auto-compress language models for microcontrollers.
From 7B parameters to 8KB RAM.
"""

__version__ = "0.1.0"
__author__ = "Aman Sachan"

from bitforge.compress.quantize import (
    LayerQuantizationResult,
    QuantizationConfig,
    QuantizationMode,
    QuantizationResult,
    Quantizer,
)
from bitforge.context import CompressedContext, ContextCompressionConfig, ContextCompressor
from bitforge.generate.c_codegen import CCodeGenerator, GeneratedFile, GeneratedProject
from bitforge.kvcache import KVCacheCompressor, KVCacheConfig, PackedKVLayer, PackedPastKeyValues, PackedTensor
from bitforge.prune import BlockPruner, PrunedLayer, PrunedModelResult, PruningConfig
from bitforge.targets.arduino import ArduinoTarget
from bitforge.targets.esp32 import ESP32Target
from bitforge.targets.stm32 import STM32Target

__all__ = [
    "Quantizer",
    "QuantizationConfig",
    "QuantizationMode",
    "QuantizationResult",
    "LayerQuantizationResult",
    "CompressedContext",
    "ContextCompressionConfig",
    "ContextCompressor",
    "KVCacheCompressor",
    "KVCacheConfig",
    "PackedTensor",
    "PackedKVLayer",
    "PackedPastKeyValues",
    "CCodeGenerator",
    "GeneratedFile",
    "GeneratedProject",
    "ESP32Target",
    "ArduinoTarget",
    "STM32Target",
    "BlockPruner",
    "PruningConfig",
    "PrunedLayer",
    "PrunedModelResult",
]
