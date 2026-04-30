"""
ESP32 target configuration.

Supports ESP32, ESP32-S2, ESP32-S3, ESP32-C3, and ESP32-C6.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class ESP32Variant(Enum):
    """ESP32 chip variants."""
    ESP32 = "esp32"
    ESP32_S2 = "esp32-s2"
    ESP32_S3 = "esp32-s3"
    ESP32_C3 = "esp32-c3"
    ESP32_C6 = "esp32-c6"


@dataclass
class ESP32Target:
    """ESP32 target configuration.
    
    Attributes:
        variant: ESP32 chip variant
        ram_kb: Available RAM in kilobytes
        flash_kb: Available Flash in kilobytes
        psram_kb: Available PSRAM (if any) in kilobytes
        has_wifi: Whether chip has WiFi
        has_ble: Whether chip has Bluetooth
        has_usb: Whether chip has USB OTG
        cores: Number of CPU cores
        frequency_mhz: CPU frequency in MHz
    """
    variant: ESP32Variant = ESP32Variant.ESP32_S3
    ram_kb: int = 512
    flash_kb: int = 4096
    psram_kb: int = 8192
    has_wifi: bool = True
    has_ble: bool = True
    has_usb: bool = True
    cores: int = 2
    frequency_mhz: int = 240
    
    # Default configurations for each variant
    VARIANTS: Dict[ESP32Variant, Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize variant-specific defaults."""
        self.VARIANTS = {
            ESP32Variant.ESP32: {
                "ram_kb": 520,
                "flash_kb": 4096,
                "psram_kb": 0,
                "has_wifi": True,
                "has_ble": True,
                "has_usb": False,
                "cores": 2,
                "frequency_mhz": 240,
            },
            ESP32Variant.ESP32_S2: {
                "ram_kb": 320,
                "flash_kb": 4096,
                "psram_kb": 0,
                "has_wifi": True,
                "has_ble": False,
                "has_usb": True,
                "cores": 1,
                "frequency_mhz": 240,
            },
            ESP32Variant.ESP32_S3: {
                "ram_kb": 512,
                "flash_kb": 4096,
                "psram_kb": 8192,
                "has_wifi": True,
                "has_ble": True,
                "has_usb": True,
                "cores": 2,
                "frequency_mhz": 240,
            },
            ESP32Variant.ESP32_C3: {
                "ram_kb": 400,
                "flash_kb": 4096,
                "psram_kb": 0,
                "has_wifi": True,
                "has_ble": True,
                "has_usb": False,
                "cores": 1,
                "frequency_mhz": 160,
            },
            ESP32Variant.ESP32_C6: {
                "ram_kb": 512,
                "flash_kb": 4096,
                "psram_kb": 0,
                "has_wifi": True,
                "has_ble": True,
                "has_usb": False,
                "cores": 1,
                "frequency_mhz": 160,
            },
        }
        
        # Apply variant defaults if not overridden
        if self.variant in self.VARIANTS:
            defaults = self.VARIANTS[self.variant]
            if self.ram_kb == 512 and defaults["ram_kb"] != 512:
                self.ram_kb = defaults["ram_kb"]
    
    def get_platformio_board(self) -> str:
        """Get PlatformIO board name.
        
        Returns:
            PlatformIO board identifier
        """
        boards = {
            ESP32Variant.ESP32: "esp32dev",
            ESP32Variant.ESP32_S2: "esp32-s2-saola-1",
            ESP32Variant.ESP32_S3: "esp32s3box",
            ESP32Variant.ESP32_C3: "esp32-c3-devkitm-1",
            ESP32Variant.ESP32_C6: "esp32-c6-devkitm-1",
        }
        return boards.get(self.variant, "esp32dev")
    
    def get_idf_target(self) -> str:
        """Get ESP-IDF target name.
        
        Returns:
            ESP-IDF target identifier
        """
        return self.variant.value
    
    def get_memory_config(self) -> Dict[str, int]:
        """Get memory configuration for model fitting.
        
        Returns:
            Dict with memory constraints in bytes
        """
        total_ram = self.ram_kb * 1024
        if self.psram_kb > 0:
            total_ram += self.psram_kb * 1024
            
        # Reserve space for system overhead
        available_ram = int(total_ram * 0.7)
        
        return {
            "total_ram_bytes": total_ram,
            "available_ram_bytes": available_ram,
            "flash_bytes": self.flash_kb * 1024,
            "max_model_size_bytes": int(self.flash_kb * 1024 * 0.8),  # 80% of flash
        }
    
    def is_compatible(self, model_size_bytes: int, ram_required_bytes: int) -> bool:
        """Check if model is compatible with this target.
        
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
    
    def get_build_flags(self) -> List[str]:
        """Get compiler build flags.
        
        Returns:
            List of build flags
        """
        flags = [
            "-DBITFORGE_TARGET_ESP32",
            f"-DBITFORGE_VARIANT_{self.variant.name.upper()}",
            f"-DBITFORGE_RAM_KB={self.ram_kb}",
            f"-DBITFORGE_FLASH_KB={self.flash_kb}",
        ]
        
        if self.has_usb:
            flags.append("-DBITFORGE_HAS_USB")
        if self.psram_kb > 0:
            flags.append("-DBITFORGE_HAS_PSRAM")
            
        return flags
    
    def get_flash_command(self, port: str = "/dev/ttyUSB0") -> str:
        """Get command to flash the model.
        
        Args:
            port: Serial port to use
            
        Returns:
            Flash command string
        """
        return f"idf.py -p {port} flash monitor"
    
    def __str__(self) -> str:
        """Get string representation."""
        return f"ESP32Target({self.variant.value}, RAM={self.ram_kb}KB, Flash={self.flash_kb}KB)"
