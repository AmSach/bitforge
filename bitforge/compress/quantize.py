"""
Bit quantization for LLM weight compression.

Supports 1-bit, 2-bit, 4-bit, and 8-bit quantization with
adaptive bit-packing based on layer sensitivity.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
import logging

logger = logging.getLogger(__name__)


class QuantizationMode(Enum):
    """Supported quantization modes."""
    BIT_1 = 1
    BIT_2 = 2
    BIT_4 = 4
    BIT_8 = 8
    ADAPTIVE = -1  # Auto-select per layer


@dataclass
class QuantizationConfig:
    """Configuration for model quantization.
    
    Attributes:
        mode: Quantization mode (1-bit, 2-bit, 4-bit, 8-bit, or adaptive)
        target_ram_kb: Target device RAM in kilobytes
        target_flash_kb: Target device Flash in kilobytes
        per_layer_sensitivity: Whether to use sensitivity analysis
        calibration_samples: Number of samples for calibration
        kv_cache_bits: Bit width for KV cache (separate from weights)
        outlier_threshold: Threshold for outlier detection (std dev)
    """
    mode: QuantizationMode = QuantizationMode.BIT_4
    target_ram_kb: int = 512
    target_flash_kb: int = 4096
    per_layer_sensitivity: bool = True
    calibration_samples: int = 128
    kv_cache_bits: int = 8
    outlier_threshold: float = 3.0
    
    def validate(self) -> Dict[str, Any]:
        """Validate configuration.
        
        Returns:
            Dict with 'valid' bool and 'errors' list
        """
        errors: List[str] = []
        
        if self.target_ram_kb < 8:
            errors.append("target_ram_kb must be at least 8 KB")
        if self.target_flash_kb < 64:
            errors.append("target_flash_kb must be at least 64 KB")
        if self.calibration_samples < 1:
            errors.append("calibration_samples must be at least 1")
        if self.kv_cache_bits not in [4, 8, 16]:
            errors.append("kv_cache_bits must be 4, 8, or 16")
        if self.outlier_threshold <= 0:
            errors.append("outlier_threshold must be positive")
            
        return {"valid": len(errors) == 0, "errors": errors}


@dataclass
class LayerQuantizationResult:
    """Result of quantizing a single layer.
    
    Attributes:
        name: Layer name
        original_shape: Original weight shape
        original_bits: Original bit width (typically 16 or 32)
        quantized_bits: Quantized bit width
        quantized_weights: Packed quantized weights
        scale: Scale factor for dequantization
        zero_point: Zero point for asymmetric quantization
        sensitivity: Sensitivity score (higher = more sensitive)
        compression_ratio: Achieved compression ratio
        mse: Mean squared error from original
    """
    name: str
    original_shape: Tuple[int, ...]
    original_bits: int
    quantized_bits: int
    quantized_weights: NDArray[np.uint8]
    scale: float
    zero_point: float
    sensitivity: float = 0.0
    compression_ratio: float = 1.0
    mse: float = 0.0


@dataclass
class QuantizationResult:
    """Result of full model quantization.
    
    Attributes:
        layers: Per-layer quantization results
        total_params: Total parameter count
        compressed_size_bytes: Compressed model size
        original_size_bytes: Original model size
        overall_compression_ratio: Overall compression achieved
        target_compatible: Whether model fits target constraints
        config: Configuration used
    """
    layers: List[LayerQuantizationResult] = field(default_factory=list)
    total_params: int = 0
    compressed_size_bytes: int = 0
    original_size_bytes: int = 0
    overall_compression_ratio: float = 1.0
    target_compatible: bool = True
    config: Optional[QuantizationConfig] = None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics.
        
        Returns:
            Dict with summary statistics
        """
        if not self.layers:
            return {}
            
        bit_distribution: Dict[int, int] = {}
        total_mse = 0.0
        
        for layer in self.layers:
            bits = layer.quantized_bits
            bit_distribution[bits] = bit_distribution.get(bits, 0) + 1
            total_mse += layer.mse
            
        avg_mse = total_mse / len(self.layers)
        
        return {
            "total_layers": len(self.layers),
            "total_params": self.total_params,
            "original_size_mb": self.original_size_bytes / (1024 * 1024),
            "compressed_size_kb": self.compressed_size_bytes / 1024,
            "compression_ratio": self.overall_compression_ratio,
            "bit_distribution": bit_distribution,
            "average_mse": avg_mse,
            "target_compatible": self.target_compatible,
        }


class Quantizer:
    """Main quantization engine for LLM compression.
    
    Handles weight quantization from FP16/FP32 to low-bit integer
    representations suitable for microcontroller deployment.
    
    Example:
        >>> config = QuantizationConfig(mode=QuantizationMode.BIT_4)
        >>> quantizer = Quantizer(config)
        >>> result = quantizer.quantize_model(model_weights)
    """
    
    def __init__(self, config: Optional[QuantizationConfig] = None):
        """Initialize quantizer with configuration.
        
        Args:
            config: Quantization configuration. Uses defaults if None.
        """
        self.config = config or QuantizationConfig()
        validation = self.config.validate()
        if not validation["valid"]:
            raise ValueError(f"Invalid config: {validation['errors']}")
            
        self._calibration_data: Optional[Dict[str, Tensor]] = None
        
    def quantize_tensor(
        self,
        tensor: Union[Tensor, NDArray[np.floating]],
        bits: int,
        scale: Optional[float] = None,
        zero_point: Optional[float] = None,
    ) -> Tuple[NDArray[np.uint8], float, float]:
        """Quantize a tensor to specified bit width.
        
        Args:
            tensor: Input tensor (float32 or float16)
            bits: Target bit width (1, 2, 4, or 8)
            scale: Optional pre-computed scale
            zero_point: Optional pre-computed zero point
            
        Returns:
            Tuple of (packed_quantized_weights, scale, zero_point)
            
        Raises:
            ValueError: If bits is not supported
        """
        if bits not in [1, 2, 4, 8]:
            raise ValueError(f"Unsupported bit width: {bits}. Must be 1, 2, 4, or 8")
            
        # Convert to numpy
        if isinstance(tensor, Tensor):
            arr = tensor.detach().cpu().numpy().astype(np.float32)
        else:
            arr = np.asarray(tensor, dtype=np.float32)
            
        # Flatten for quantization
        original_shape = arr.shape
        flat = arr.flatten()
        
        # Handle outliers
        if self.config.outlier_threshold > 0:
            flat = self._handle_outliers(flat, self.config.outlier_threshold)
            
        # Compute scale and zero point if not provided
        if scale is None or zero_point is None:
            scale, zero_point = self._compute_scale_zeropoint(flat, bits)
            
        # Quantize
        qmin = 0
        qmax = (1 << bits) - 1
        
        # Scale to quantized range
        quantized = np.round((flat - zero_point) / scale)
        quantized = np.clip(quantized, qmin, qmax).astype(np.int32)
        
        # Pack bits
        packed = self._pack_bits(quantized, bits)
        
        logger.debug(
            f"Quantized {original_shape} to {bits}-bit: "
            f"scale={scale:.6f}, zp={zero_point:.6f}"
        )
        
        return packed, scale, zero_point
    
    def dequantize_tensor(
        self,
        packed: NDArray[np.uint8],
        bits: int,
        scale: float,
        zero_point: float,
        shape: Tuple[int, ...],
    ) -> NDArray[np.float32]:
        """Dequantize packed weights back to float.
        
        Args:
            packed: Packed quantized weights
            bits: Bit width used for quantization
            scale: Scale factor
            zero_point: Zero point
            shape: Original tensor shape
            
        Returns:
            Dequantized float array
        """
        # Unpack bits
        quantized = self._unpack_bits(packed, bits, int(np.prod(shape)))
        
        # Dequantize
        dequantized = (quantized.astype(np.float32) * scale) + zero_point
        
        return dequantized.reshape(shape)
    
    def compute_sensitivity(
        self,
        tensor: Union[Tensor, NDArray[np.floating]],
        bits: int,
        num_samples: int = 100,
    ) -> float:
        """Compute sensitivity score for a layer.
        
        Higher sensitivity means the layer is more affected by quantization
        and should retain higher bit width.
        
        Args:
            tensor: Layer weights
            bits: Bit width to test
            num_samples: Number of random perturbations
            
        Returns:
            Sensitivity score (higher = more sensitive)
        """
        if isinstance(tensor, Tensor):
            arr = tensor.detach().cpu().numpy().astype(np.float32)
        else:
            arr = np.asarray(tensor, dtype=np.float32)
            
        # Quantize and dequantize
        packed, scale, zp = self.quantize_tensor(arr, bits)
        dequantized = self.dequantize_tensor(packed, bits, scale, zp, arr.shape)
        
        # Compute MSE
        mse = np.mean((arr - dequantized) ** 2)
        
        # Compute gradient sensitivity
        grad = np.gradient(arr)
        grad_var = np.var(grad)
        
        # Combined sensitivity score
        sensitivity = mse * (1 + grad_var)
        
        return float(sensitivity)
    
    def quantize_layer(
        self,
        name: str,
        weights: Union[Tensor, NDArray[np.floating]],
        bits: Optional[int] = None,
    ) -> LayerQuantizationResult:
        """Quantize a single layer with optional sensitivity analysis.
        
        Args:
            name: Layer name
            weights: Layer weights
            bits: Override bit width (uses config if None)
            
        Returns:
            LayerQuantizationResult with packed weights and metadata
        """
        if isinstance(weights, Tensor):
            arr = weights.detach().cpu().numpy().astype(np.float32)
            original_bits = 16 if weights.dtype == torch.float16 else 32
        else:
            arr = np.asarray(weights, dtype=np.float32)
            original_bits = 32
            
        original_shape = arr.shape
        num_params = int(np.prod(original_shape))
        
        # Determine bit width
        if bits is None:
            if self.config.mode == QuantizationMode.ADAPTIVE:
                bits = self._select_adaptive_bits(name, arr)
            else:
                bits = self.config.mode.value
                
        # Compute sensitivity
        sensitivity = self.compute_sensitivity(arr, bits) if self.config.per_layer_sensitivity else 0.0
        
        # Quantize
        packed, scale, zp = self.quantize_tensor(arr, bits)
        
        # Compute metrics
        dequantized = self.dequantize_tensor(packed, bits, scale, zp, original_shape)
        mse = float(np.mean((arr - dequantized) ** 2))
        
        original_bytes = num_params * (original_bits // 8)
        compressed_bytes = packed.nbytes
        compression_ratio = original_bytes / compressed_bytes if compressed_bytes > 0 else 1.0
        
        return LayerQuantizationResult(
            name=name,
            original_shape=original_shape,
            original_bits=original_bits,
            quantized_bits=bits,
            quantized_weights=packed,
            scale=scale,
            zero_point=zp,
            sensitivity=sensitivity,
            compression_ratio=compression_ratio,
            mse=mse,
        )
    
    def quantize_model(
        self,
        weights: Dict[str, Union[Tensor, NDArray[np.floating]]],
    ) -> QuantizationResult:
        """Quantize all weights in a model.
        
        Args:
            weights: Dict mapping layer names to weight tensors
            
        Returns:
            QuantizationResult with all layer results and summary
        """
        layers: List[LayerQuantizationResult] = []
        total_params = 0
        original_size = 0
        compressed_size = 0
        
        for name, tensor in weights.items():
            result = self.quantize_layer(name, tensor)
            layers.append(result)
            
            num_params = int(np.prod(result.original_shape))
            total_params += num_params
            original_size += num_params * (result.original_bits // 8)
            compressed_size += result.quantized_weights.nbytes
            
        overall_compression = original_size / compressed_size if compressed_size > 0 else 1.0
        
        # Check target compatibility
        ram_kb = self.config.target_ram_kb
        flash_kb = self.config.target_flash_kb
        compressed_kb = compressed_size / 1024
        
        target_compatible = compressed_kb <= flash_kb
        
        return QuantizationResult(
            layers=layers,
            total_params=total_params,
            compressed_size_bytes=compressed_size,
            original_size_bytes=original_size,
            overall_compression_ratio=overall_compression,
            target_compatible=target_compatible,
            config=self.config,
        )
    
    def _compute_scale_zeropoint(
        self,
        arr: NDArray[np.float32],
        bits: int,
    ) -> Tuple[float, float]:
        """Compute scale and zero point for quantization.
        
        Args:
            arr: Flattened float array
            bits: Target bit width
            
        Returns:
            Tuple of (scale, zero_point)
        """
        qmin = 0
        qmax = (1 << bits) - 1
        
        # Symmetric quantization for 1-bit and 2-bit
        if bits <= 2:
            abs_max = np.max(np.abs(arr))
            scale = abs_max / ((qmax + 1) / 2) if abs_max > 0 else 1.0
            zero_point = 0.0
        else:
            # Asymmetric quantization for 4-bit and 8-bit
            min_val = np.min(arr)
            max_val = np.max(arr)
            
            scale = (max_val - min_val) / (qmax - qmin) if max_val > min_val else 1.0
            zero_point = min_val
            
        return float(scale), float(zero_point)
    
    def _handle_outliers(
        self,
        arr: NDArray[np.float32],
        threshold: float,
    ) -> NDArray[np.float32]:
        """Clip outliers based on standard deviation threshold.
        
        Args:
            arr: Input array
            threshold: Number of standard deviations
            
        Returns:
            Clipped array
        """
        mean = np.mean(arr)
        std = np.std(arr)
        if std == 0:
            return arr
            
        lower = mean - threshold * std
        upper = mean + threshold * std
        
        return np.clip(arr, lower, upper)
    
    def _pack_bits(
        self,
        quantized: NDArray[np.int32],
        bits: int,
    ) -> NDArray[np.uint8]:
        """Pack quantized values into bytes.
        
        Args:
            quantized: Quantized integer values
            bits: Bit width of each value
            
        Returns:
            Packed uint8 array
        """
        if bits == 8:
            return quantized.astype(np.uint8)
            
        # Calculate packed size
        num_values = len(quantized)
        values_per_byte = 8 // bits
        packed_size = (num_values + values_per_byte - 1) // values_per_byte
        
        packed = np.zeros(packed_size, dtype=np.uint8)
        
        for i, val in enumerate(quantized):
            byte_idx = i // values_per_byte
            bit_offset = (i % values_per_byte) * bits
            packed[byte_idx] |= (val << bit_offset)
            
        return packed
    
    def _unpack_bits(
        self,
        packed: NDArray[np.uint8],
        bits: int,
        num_values: int,
    ) -> NDArray[np.int32]:
        """Unpack quantized values from packed bytes.
        
        Args:
            packed: Packed uint8 array
            bits: Bit width of each value
            num_values: Number of values to unpack
            
        Returns:
            Unpacked integer array
        """
        if bits == 8:
            return packed[:num_values].astype(np.int32)
            
        values_per_byte = 8 // bits
        mask = (1 << bits) - 1
        
        unpacked = np.zeros(num_values, dtype=np.int32)
        
        for i in range(num_values):
            byte_idx = i // values_per_byte
            bit_offset = (i % values_per_byte) * bits
            unpacked[i] = (packed[byte_idx] >> bit_offset) & mask
            
        return unpacked
    
    def _select_adaptive_bits(self, name: str, arr: NDArray[np.float32]) -> int:
        """Select optimal bit width based on layer analysis.
        
        Args:
            name: Layer name
            arr: Layer weights
            
        Returns:
            Selected bit width
        """
        # Layers that are more sensitive
        sensitive_keywords = ["embed", "head", "attn", "q_proj", "k_proj", "v_proj"]
        is_sensitive = any(kw in name.lower() for kw in sensitive_keywords)
        
        # Test different bit widths
        sensitivities = {}
        for bits in [1, 2, 4, 8]:
            sens = self.compute_sensitivity(arr, bits)
            sensitivities[bits] = sens
            
        # Select based on sensitivity and layer type
        if is_sensitive:
            # Prefer higher bits for sensitive layers
            for bits in [4, 2, 1]:
                if sensitivities[bits] < 0.01:  # Threshold for acceptable loss
                    return bits
            return 8
        else:
            # Can use lower bits for less sensitive layers
            for bits in [1, 2, 4]:
                if sensitivities[bits] < 0.1:
                    return bits
            return 4
