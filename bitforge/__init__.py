"""
BitForge - Shrink any LLM to fit in your pocket.

Auto-compress language models for microcontrollers.
From 7B parameters to 8KB RAM.
"""

__version__ = "0.1.0"
__author__ = "Aman Sachan"

from bitforge.compress.quantize import Quantizer, QuantizationConfig
from bitforge.generate.c_codegen import CCodeGenerator
from bitforge.targets.esp32 import ESP32Target
from bitforge.targets.arduino import ArduinoTarget
from bitforge.targets.stm32 import STM32Target

__all__ = [
    "Quantizer",
    "QuantizationConfig", 
    "CCodeGenerator",
    "ESP32Target",
    "ArduinoTarget",
    "STM32Target",
]
