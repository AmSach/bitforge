"""Target configuration modules."""

from bitforge.targets.esp32 import ESP32Target
from bitforge.targets.arduino import ArduinoTarget
from bitforge.targets.stm32 import STM32Target

__all__ = ["ESP32Target", "ArduinoTarget", "STM32Target"]
