"""
Arduino/AVR target configuration.

Supports Arduino Uno, Nano, Mega, and compatible AVR boards.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class ArduinoVariant(Enum):
    """Arduino board variants."""
    UNO = "uno"
    NANO = "nano"
    MEGA = "mega"
    MEGA2560 = "mega2560"
    LEONARDO = "leonardo"
    MICRO = "micro"


@dataclass
class ArduinoTarget:
    """Arduino/AVR target configuration.
    
    Arduino boards have very limited resources, requiring
    extreme quantization (1-bit or 2-bit) for most models.
    
    Attributes:
        variant: Arduino board variant
        ram_bytes: Available SRAM in bytes (not KB!)
        flash_bytes: Available Flash in bytes
        clock_mhz: CPU clock speed in MHz
        has_usb_serial: Whether board has USB-serial
    """
    variant: ArduinoVariant = ArduinoVariant.MEGA2560
    ram_bytes: int = 8192  # 8KB for Mega
    flash_bytes: int = 262144  # 256KB for Mega
    clock_mhz: int = 16
    has_usb_serial: bool = True
    
    # Default configurations for each variant
    VARIANTS: Dict[ArduinoVariant, Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize variant-specific defaults."""
        self.VARIANTS = {
            ArduinoVariant.UNO: {
                "ram_bytes": 2048,  # 2KB
                "flash_bytes": 32768,  # 32KB
                "clock_mhz": 16,
                "has_usb_serial": False,
            },
            ArduinoVariant.NANO: {
                "ram_bytes": 2048,
                "flash_bytes": 32768,
                "clock_mhz": 16,
                "has_usb_serial": False,
            },
            ArduinoVariant.MEGA: {
                "ram_bytes": 8192,
                "flash_bytes": 131072,  # 128KB
                "clock_mhz": 16,
                "has_usb_serial": False,
            },
            ArduinoVariant.MEGA2560: {
                "ram_bytes": 8192,
                "flash_bytes": 262144,  # 256KB
                "clock_mhz": 16,
                "has_usb_serial": False,
            },
            ArduinoVariant.LEONARDO: {
                "ram_bytes": 2560,
                "flash_bytes": 28672,  # 28KB available
                "clock_mhz": 16,
                "has_usb_serial": True,
            },
            ArduinoVariant.MICRO: {
                "ram_bytes": 2560,
                "flash_bytes": 28672,
                "clock_mhz": 16,
                "has_usb_serial": True,
            },
        }
        
        # Apply variant defaults
        if self.variant in self.VARIANTS:
            defaults = self.VARIANTS[self.variant]
            if self.ram_bytes == 8192 and defaults.get("ram_bytes") != 8192:
                self.ram_bytes = defaults.get("ram_bytes", 8192)
    
    def get_platformio_board(self) -> str:
        """Get PlatformIO board name.
        
        Returns:
            PlatformIO board identifier
        """
        boards = {
            ArduinoVariant.UNO: "uno",
            ArduinoVariant.NANO: "nanoatmega328",
            ArduinoVariant.MEGA: "megaatmega1280",
            ArduinoVariant.MEGA2560: "megaatmega2560",
            ArduinoVariant.LEONARDO: "leonardo",
            ArduinoVariant.MICRO: "micro",
        }
        return boards.get(self.variant, "uno")
    
    def get_arduino_board(self) -> str:
        """Get Arduino IDE board name.
        
        Returns:
            Arduino board identifier
        """
        boards = {
            ArduinoVariant.UNO: "arduino:avr:uno",
            ArduinoVariant.NANO: "arduino:avr:nano:cpu=atmega328",
            ArduinoVariant.MEGA: "arduino:avr:mega:cpu=atmega1280",
            ArduinoVariant.MEGA2560: "arduino:avr:mega:cpu=atmega2560",
            ArduinoVariant.LEONARDO: "arduino:avr:leonardo",
            ArduinoVariant.MICRO: "arduino:avr:micro",
        }
        return boards.get(self.variant, "arduino:avr:uno")
    
    def get_memory_config(self) -> Dict[str, int]:
        """Get memory configuration for model fitting.
        
        Returns:
            Dict with memory constraints in bytes
        """
        # Reserve 1KB for stack and heap overhead
        available_ram = max(self.ram_bytes - 1024, 512)
        
        # Reserve 4KB for program code
        available_flash = max(self.flash_bytes - 4096, 8192)
        
        return {
            "total_ram_bytes": self.ram_bytes,
            "available_ram_bytes": available_ram,
            "flash_bytes": self.flash_bytes,
            "available_flash_bytes": available_flash,
            "max_model_size_bytes": int(available_flash * 0.9),
        }
    
    def is_compatible(self, model_size_bytes: int, ram_required_bytes: int) -> bool:
        """Check if model is compatible with this target.
        
        Arduino boards are extremely constrained, so this
        check is very strict.
        
        Args:
            model_size_bytes: Size of model weights
            ram_required_bytes: RAM needed for inference
            
        Returns:
            True if model fits in constraints
        """
        memory = self.get_memory_config()
        
        fits_flash = model_size_bytes <= memory["max_model_size_bytes"]
        fits_ram = ram_required_bytes <= memory["available_ram_bytes"]
        
        return fits_flash and fits_ram
    
    def get_recommended_bits(self) -> int:
        """Get recommended quantization bit width.
        
        Arduino boards need extreme quantization.
        
        Returns:
            Recommended bit width (1 or 2)
        """
        if self.ram_bytes <= 2048:
            return 1  # 1-bit only for Uno/Nano
        elif self.ram_bytes <= 4096:
            return 1  # Still 1-bit
        else:
            return 2  # 2-bit for Mega
    
    def get_build_flags(self) -> List[str]:
        """Get compiler build flags.
        
        Returns:
            List of build flags
        """
        flags = [
            "-DBITFORGE_TARGET_ARDUINO",
            f"-DBITFORGE_VARIANT_{self.variant.name.upper()}",
            f"-DBITFORGE_RAM_BYTES={self.ram_bytes}",
            f"-DBITFORGE_FLASH_BYTES={self.flash_bytes}",
            "-Os",  # Optimize for size
            "-fno-inline-functions",  # Reduce code size
        ]
        
        if self.clock_mhz <= 8:
            flags.append("-DBITFORGE_SLOW_CPU")
            
        return flags
    
    def get_flash_command(self, port: str = "/dev/ttyUSB0") -> str:
        """Get command to flash the model.
        
        Args:
            port: Serial port to use
            
        Returns:
            Flash command string
        """
        board = self.get_platformio_board()
        return f"arduino-cli upload -p {port} -b {self.get_arduino_board()} --input-dir ."
    
    def __str__(self) -> str:
        """Get string representation."""
        ram_kb = self.ram_bytes / 1024
        flash_kb = self.flash_bytes / 1024
        return f"ArduinoTarget({self.variant.value}, RAM={ram_kb:.0f}KB, Flash={flash_kb:.0f}KB)"
