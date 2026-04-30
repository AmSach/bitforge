"""
STM32 target configuration.

Supports STM32F4, STM32F7, STM32H7, and other ARM Cortex-M series.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class STM32Family(Enum):
    """STM32 chip families."""
    STM32F1 = "stm32f1"
    STM32F4 = "stm32f4"
    STM32F7 = "stm32f7"
    STM32H7 = "stm32h7"
    STM32L4 = "stm32l4"


class STM32Core(Enum):
    """ARM Cortex core types."""
    CORTEX_M0 = "cortex-m0"
    CORTEX_M0PLUS = "cortex-m0plus"
    CORTEX_M3 = "cortex-m3"
    CORTEX_M4 = "cortex-m4"
    CORTEX_M7 = "cortex-m7"


@dataclass
class STM32Target:
    """STM32 target configuration.
    
    STM32 boards offer good balance of resources for
    running quantized LLMs with moderate compression.
    
    Attributes:
        family: STM32 chip family
        core: ARM Cortex core type
        ram_kb: Available RAM in kilobytes
        flash_kb: Available Flash in kilobytes
        has_fpu: Whether chip has FPU
        frequency_mhz: CPU frequency in MHz
        has_hardware_crypto: Whether chip has crypto accelerator
    """
    family: STM32Family = STM32Family.STM32F4
    core: STM32Core = STM32Core.CORTEX_M4
    ram_kb: int = 192
    flash_kb: int = 1024
    has_fpu: bool = True
    frequency_mhz: int = 168
    has_hardware_crypto: bool = False
    
    # Default configurations for each family
    FAMILIES: Dict[STM32Family, Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize family-specific defaults."""
        self.FAMILIES = {
            STM32Family.STM32F1: {
                "core": STM32Core.CORTEX_M3,
                "ram_kb": 96,
                "flash_kb": 512,
                "has_fpu": False,
                "frequency_mhz": 72,
                "has_hardware_crypto": False,
            },
            STM32Family.STM32F4: {
                "core": STM32Core.CORTEX_M4,
                "ram_kb": 192,
                "flash_kb": 1024,
                "has_fpu": True,
                "frequency_mhz": 168,
                "has_hardware_crypto": True,
            },
            STM32Family.STM32F7: {
                "core": STM32Core.CORTEX_M7,
                "ram_kb": 512,
                "flash_kb": 2048,
                "has_fpu": True,
                "frequency_mhz": 216,
                "has_hardware_crypto": True,
            },
            STM32Family.STM32H7: {
                "core": STM32Core.CORTEX_M7,
                "ram_kb": 1024,
                "flash_kb": 2048,
                "has_fpu": True,
                "frequency_mhz": 480,
                "has_hardware_crypto": True,
            },
            STM32Family.STM32L4: {
                "core": STM32Core.CORTEX_M4,
                "ram_kb": 128,
                "flash_kb": 1024,
                "has_fpu": True,
                "frequency_mhz": 80,
                "has_hardware_crypto": True,
            },
        }
        
        # Apply family defaults
        if self.family in self.FAMILIES:
            defaults = self.FAMILIES[self.family]
            if self.core == STM32Core.CORTEX_M4 and defaults.get("core") != STM32Core.CORTEX_M4:
                self.core = defaults.get("core", STM32Core.CORTEX_M4)
    
    def get_platformio_board(self) -> str:
        """Get PlatformIO board name.
        
        Returns:
            PlatformIO board identifier
        """
        boards = {
            STM32Family.STM32F1: "genericSTM32F103RE",
            STM32Family.STM32F4: "genericSTM32F407VET6",
            STM32Family.STM32F7: "genericSTM32F746ZG",
            STM32Family.STM32H7: "genericSTM32H743ZI",
            STM32Family.STM32L4: "genericSTM32L476RG",
        }
        return boards.get(self.family, "genericSTM32F407VET6")
    
    def get_cpu_flags(self) -> List[str]:
        """Get CPU-specific compiler flags.
        
        Returns:
            List of CPU flags
        """
        flags_map = {
            STM32Core.CORTEX_M0: ["-mcpu=cortex-m0", "-mthumb"],
            STM32Core.CORTEX_M0PLUS: ["-mcpu=cortex-m0plus", "-mthumb"],
            STM32Core.CORTEX_M3: ["-mcpu=cortex-m3", "-mthumb"],
            STM32Core.CORTEX_M4: ["-mcpu=cortex-m4", "-mthumb", "-mfloat-abi=hard", "-mfpu=fpv4-sp-d16"],
            STM32Core.CORTEX_M7: ["-mcpu=cortex-m7", "-mthumb", "-mfloat-abi=hard", "-mfpu=fpv5-d16"],
        }
        return flags_map.get(self.core, ["-mcpu=cortex-m4", "-mthumb"])
    
    def get_memory_config(self) -> Dict[str, int]:
        """Get memory configuration for model fitting.
        
        Returns:
            Dict with memory constraints in bytes
        """
        total_ram = self.ram_kb * 1024
        
        # Reserve 16KB for stack and heap
        available_ram = max(total_ram - 16384, 32768)
        
        # Reserve 64KB for program code
        available_flash = max(self.flash_kb * 1024 - 65536, 65536)
        
        return {
            "total_ram_bytes": total_ram,
            "available_ram_bytes": available_ram,
            "flash_bytes": self.flash_kb * 1024,
            "available_flash_bytes": available_flash,
            "max_model_size_bytes": int(available_flash * 0.85),
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
    
    def get_recommended_bits(self) -> int:
        """Get recommended quantization bit width.
        
        Returns:
            Recommended bit width
        """
        if self.ram_kb <= 128:
            return 2  # 2-bit for smaller STM32
        elif self.ram_kb <= 256:
            return 4  # 4-bit for mid-range
        else:
            return 4  # 4-bit for larger STM32
    
    def get_build_flags(self) -> List[str]:
        """Get compiler build flags.
        
        Returns:
            List of build flags
        """
        flags = [
            "-DBITFORGE_TARGET_STM32",
            f"-DBITFORGE_FAMILY_{self.family.name.upper()}",
            f"-DBITFORGE_CORE_{self.core.name.upper()}",
            f"-DBITFORGE_RAM_KB={self.ram_kb}",
            f"-DBITFORGE_FLASH_KB={self.flash_kb}",
        ]
        
        flags.extend(self.get_cpu_flags())
        
        if self.has_fpu:
            flags.append("-DBITFORGE_HAS_FPU")
        if self.has_hardware_crypto:
            flags.append("-DBITFORGE_HAS_CRYPTO")
            
        return flags
    
    def get_flash_command(self, port: str = "/dev/ttyUSB0") -> str:
        """Get command to flash the model.
        
        Args:
            port: Serial port to use
            
        Returns:
            Flash command string
        """
        board = self.get_platformio_board()
        return f"platformio run --target upload --upload-port {port} -e {board}"
    
    def __str__(self) -> str:
        """Get string representation."""
        return f"STM32Target({self.family.value}, Core={self.core.value}, RAM={self.ram_kb}KB, Flash={self.flash_kb}KB)"
