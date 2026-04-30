"""Tests for BitForge quantization module."""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
import torch
import sys

sys.path.insert(0, '/home/workspace/bitforge/src')

from bitforge import (
    QuantizationConfig,
    QuantizationMode,
    Quantizer,
    QuantizationResult,
    LayerQuantizationResult,
)


class TestQuantizationConfig:
    """Tests for QuantizationConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = QuantizationConfig()
        assert config.mode == QuantizationMode.BIT_4
        assert config.target_ram_kb == 512
        assert config.target_flash_kb == 4096
        assert config.per_layer_sensitivity is True
        assert config.calibration_samples == 128
        assert config.kv_cache_bits == 8
        assert config.outlier_threshold == 3.0
    
    def test_validation_valid(self):
        """Test validation of valid config."""
        config = QuantizationConfig()
        result = config.validate()
        assert result["valid"] is True
        assert len(result["errors"]) == 0
    
    def test_validation_invalid_ram(self):
        """Test validation catches invalid RAM."""
        config = QuantizationConfig(target_ram_kb=4)
        result = config.validate()
        assert result["valid"] is False
        assert any("target_ram_kb" in e for e in result["errors"])
    
    def test_validation_invalid_flash(self):
        """Test validation catches invalid Flash."""
        config = QuantizationConfig(target_flash_kb=32)
        result = config.validate()
        assert result["valid"] is False


class TestQuantizer:
    """Tests for Quantizer class."""
    
    @pytest.fixture
    def quantizer(self):
        return Quantizer()
    
    @pytest.fixture
    def sample_weights(self):
        return np.random.randn(100, 50).astype(np.float32) * 0.5
    
    def test_init_default_config(self, quantizer):
        assert quantizer.config is not None
        assert quantizer.config.mode == QuantizationMode.BIT_4
    
    def test_init_custom_config(self):
        config = QuantizationConfig(mode=QuantizationMode.BIT_2)
        quantizer = Quantizer(config)
        assert quantizer.config.mode == QuantizationMode.BIT_2
    
    def test_quantize_tensor_8bit(self, quantizer, sample_weights):
        packed, scale, zp = quantizer.quantize_tensor(sample_weights, bits=8)
        assert packed.dtype == np.uint8
        assert packed.ndim == 1
        assert scale > 0
    
    def test_quantize_tensor_4bit(self, quantizer, sample_weights):
        packed, scale, zp = quantizer.quantize_tensor(sample_weights, bits=4)
        assert packed.dtype == np.uint8
        expected_size = (sample_weights.size + 1) // 2
        assert len(packed) == expected_size
    
    def test_quantize_tensor_2bit(self, quantizer, sample_weights):
        packed, scale, zp = quantizer.quantize_tensor(sample_weights, bits=2)
        assert packed.dtype == np.uint8
        expected_size = (sample_weights.size + 3) // 4
        assert len(packed) == expected_size
    
    def test_quantize_tensor_1bit(self, quantizer, sample_weights):
        packed, scale, zp = quantizer.quantize_tensor(sample_weights, bits=1)
        assert packed.dtype == np.uint8
        expected_size = (sample_weights.size + 7) // 8
        assert len(packed) == expected_size
    
    def test_quantize_invalid_bits(self, quantizer, sample_weights):
        with pytest.raises(ValueError, match="Unsupported bit width"):
            quantizer.quantize_tensor(sample_weights, bits=3)
    
    def test_dequantize_8bit(self, quantizer, sample_weights):
        packed, scale, zp = quantizer.quantize_tensor(sample_weights, bits=8)
        dequantized = quantizer.dequantize_tensor(packed, 8, scale, zp, sample_weights.shape)
        assert dequantized.shape == sample_weights.shape
        correlation = np.corrcoef(sample_weights.flatten(), dequantized.flatten())[0, 1]
        assert correlation > 0.99
    
    def test_dequantize_4bit(self, quantizer, sample_weights):
        packed, scale, zp = quantizer.quantize_tensor(sample_weights, bits=4)
        dequantized = quantizer.dequantize_tensor(packed, 4, scale, zp, sample_weights.shape)
        assert dequantized.shape == sample_weights.shape
        correlation = np.corrcoef(sample_weights.flatten(), dequantized.flatten())[0, 1]
        assert correlation > 0.95
    
    def test_dequantize_2bit(self, quantizer, sample_weights):
        packed, scale, zp = quantizer.quantize_tensor(sample_weights, bits=2)
        dequantized = quantizer.dequantize_tensor(packed, 2, scale, zp, sample_weights.shape)
        assert dequantized.shape == sample_weights.shape
        correlation = np.corrcoef(sample_weights.flatten(), dequantized.flatten())[0, 1]
        # 2-bit is quite lossy, 0.6+ correlation is acceptable
        assert correlation > 0.6
    
    def test_dequantize_1bit(self, quantizer, sample_weights):
        packed, scale, zp = quantizer.quantize_tensor(sample_weights, bits=1)
        dequantized = quantizer.dequantize_tensor(packed, 1, scale, zp, sample_weights.shape)
        assert dequantized.shape == sample_weights.shape
        # 1-bit is very lossy, sign preservation check
        sign_match = np.mean((sample_weights > 0) == (dequantized > 0))
        assert sign_match > 0.55  # Most signs should match
    
    def test_compute_sensitivity(self, quantizer, sample_weights):
        sensitivity_8 = quantizer.compute_sensitivity(sample_weights, bits=8)
        sensitivity_4 = quantizer.compute_sensitivity(sample_weights, bits=4)
        sensitivity_1 = quantizer.compute_sensitivity(sample_weights, bits=1)
        assert sensitivity_1 > sensitivity_4
        assert sensitivity_4 > sensitivity_8
    
    def test_quantize_layer(self, quantizer, sample_weights):
        result = quantizer.quantize_layer("test_layer", sample_weights)
        assert isinstance(result, LayerQuantizationResult)
        assert result.name == "test_layer"
        assert result.original_shape == sample_weights.shape
        assert result.quantized_bits == 4
        assert result.compression_ratio > 1.0
    
    def test_quantize_layer_custom_bits(self, quantizer, sample_weights):
        result = quantizer.quantize_layer("test_layer", sample_weights, bits=2)
        assert result.quantized_bits == 2
    
    def test_quantize_model(self, quantizer):
        weights = {
            "layer1.weight": np.random.randn(100, 50).astype(np.float32),
            "layer1.bias": np.random.randn(50).astype(np.float32),
            "layer2.weight": np.random.randn(50, 25).astype(np.float32),
        }
        result = quantizer.quantize_model(weights)
        assert isinstance(result, QuantizationResult)
        assert len(result.layers) == 3
        assert result.total_params == 100*50 + 50 + 50*25
        assert result.overall_compression_ratio > 1.0
    
    def test_quantize_model_target_compatible(self, quantizer):
        small_weights = {"layer.weight": np.random.randn(10, 10).astype(np.float32)}
        result = quantizer.quantize_model(small_weights)
        assert result.target_compatible is True
    
    def test_quantize_torch_tensor(self, quantizer):
        tensor = torch.randn(100, 50)
        packed, scale, zp = quantizer.quantize_tensor(tensor, bits=4)
        assert packed.dtype == np.uint8
    
    def test_pack_unpack_roundtrip(self, quantizer):
        for bits in [1, 2, 4, 8]:
            values = np.random.randint(0, (1 << bits), size=100)
            packed = quantizer._pack_bits(values, bits)
            unpacked = quantizer._unpack_bits(packed, bits, len(values))
            np.testing.assert_array_equal(values, unpacked)


class TestQuantizationResult:
    """Tests for QuantizationResult."""
    
    def test_empty_result(self):
        result = QuantizationResult()
        summary = result.get_summary()
        assert summary == {}
    
    def test_result_summary(self):
        layer1 = LayerQuantizationResult(
            name="layer1",
            original_shape=(100, 50),
            original_bits=32,
            quantized_bits=4,
            quantized_weights=np.zeros(2500, dtype=np.uint8),
            scale=0.1,
            zero_point=0.0,
            mse=0.001,
        )
        result = QuantizationResult(
            layers=[layer1],
            total_params=5000,
            compressed_size_bytes=2500,
            original_size_bytes=20000,
            overall_compression_ratio=8.0,
        )
        summary = result.get_summary()
        assert summary["total_layers"] == 1
        assert summary["bit_distribution"] == {4: 1}


class TestLayerQuantizationResult:
    """Tests for LayerQuantizationResult."""
    
    def test_result_creation(self):
        result = LayerQuantizationResult(
            name="test",
            original_shape=(100, 50),
            original_bits=16,
            quantized_bits=4,
            quantized_weights=np.zeros(2500, dtype=np.uint8),
            scale=0.1,
            zero_point=0.0,
        )
        assert result.name == "test"
        assert result.sensitivity == 0.0
        assert result.compression_ratio == 1.0


class TestAdaptiveQuantization:
    """Tests for adaptive quantization."""
    
    def test_adaptive_mode_selection(self):
        config = QuantizationConfig(mode=QuantizationMode.ADAPTIVE)
        quantizer = Quantizer(config)
        sensitive_weights = np.random.randn(1000, 256).astype(np.float32) * 0.01
        bits = quantizer._select_adaptive_bits("wte.weight", sensitive_weights)
        assert bits >= 1
