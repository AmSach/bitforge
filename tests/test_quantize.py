"""Tests for BitForge quantization module."""

import pytest
import numpy as np
import torch

from bitforge.compress.quantize import (
    QuantizationConfig,
    QuantizationMode,
    Quantizer,
    QuantizationResult,
    LayerQuantizationResult,
)


class TestQuantizationConfig:
    def test_default_config(self):
        config = QuantizationConfig()
        assert config.mode == QuantizationMode.BIT_4
        assert config.target_ram_kb == 512
        assert config.target_flash_kb == 4096
        assert config.per_layer_sensitivity is True
        assert config.calibration_samples == 128
        assert config.kv_cache_bits == 8
        assert config.outlier_threshold == 3.0

    def test_validation_valid(self):
        config = QuantizationConfig()
        result = config.validate()
        assert result["valid"] is True
        assert len(result["errors"]) == 0


class TestQuantizer:
    @pytest.fixture
    def quantizer(self):
        return Quantizer()

    @pytest.fixture
    def sample_weights(self):
        return np.random.randn(100, 50).astype(np.float32) * 0.5

    def test_quantize_tensor_4bit(self, quantizer, sample_weights):
        packed, scale, zp = quantizer.quantize_tensor(sample_weights, bits=4)
        assert packed.dtype == np.uint8
        assert packed.ndim == 1
        assert scale > 0

    def test_quantize_tensor_2bit(self, quantizer, sample_weights):
        packed, scale, zp = quantizer.quantize_tensor(sample_weights, bits=2)
        assert packed.dtype == np.uint8
        assert len(packed) == (sample_weights.size + 3) // 4

    def test_quantize_invalid_bits(self, quantizer, sample_weights):
        with pytest.raises(ValueError):
            quantizer.quantize_tensor(sample_weights, bits=3)

    def test_roundtrip(self, quantizer, sample_weights):
        packed, scale, zp = quantizer.quantize_tensor(sample_weights, bits=4)
        dequantized = quantizer.dequantize_tensor(packed, 4, scale, zp, sample_weights.shape)
        assert dequantized.shape == sample_weights.shape


class TestQuantizationResult:
    def test_summary(self):
        layer = LayerQuantizationResult(
            name="layer1",
            original_shape=(10, 10),
            original_bits=32,
            quantized_bits=4,
            quantized_weights=np.zeros(50, dtype=np.uint8),
            scale=0.1,
            zero_point=0.0,
            mse=0.001,
        )
        result = QuantizationResult(layers=[layer], total_params=100, compressed_size_bytes=50, original_size_bytes=400)
        summary = result.get_summary()
        assert summary["total_layers"] == 1
        assert summary["bit_distribution"] == {4: 1}



def test_imports_root_api():
    from bitforge import ContextCompressor, ContextCompressionConfig, BlockPruner, PruningConfig
    assert ContextCompressor is not None
    assert BlockPruner is not None
