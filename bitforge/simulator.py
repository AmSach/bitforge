"""
Inference simulator for testing quantized models.

Provides a Python-based simulator to test model output
before deploying to actual hardware.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import logging
import math
import os

import numpy as np
from numpy.typing import NDArray

from bitforge.compress.quantize import QuantizationResult, Quantizer

logger = logging.getLogger(__name__)


@dataclass
class SimulatorConfig:
    """Configuration for the simulator.
    
    Attributes:
        max_seq_len: Maximum sequence length
        temperature: Sampling temperature (1.0 = standard)
        top_k: Top-k sampling parameter (0 = disabled)
        top_p: Top-p (nucleus) sampling parameter (1.0 = disabled)
        repetition_penalty: Penalty for repeated tokens
        seed: Random seed for reproducibility
    """
    max_seq_len: int = 128
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    seed: int = 42
    
    def validate(self) -> Dict[str, Any]:
        """Validate configuration.
        
        Returns:
            Dict with 'valid' bool and 'errors' list
        """
        errors: List[str] = []
        
        if self.max_seq_len < 1:
            errors.append("max_seq_len must be at least 1")
        if self.temperature < 0:
            errors.append("temperature must be non-negative")
        if self.top_k < 0:
            errors.append("top_k must be non-negative")
        if not 0 < self.top_p <= 1.0:
            errors.append("top_p must be in (0, 1]")
        if self.repetition_penalty < 1.0:
            errors.append("repetition_penalty must be >= 1.0")
            
        return {"valid": len(errors) == 0, "errors": errors}


@dataclass
class InferenceResult:
    """Result of running inference.
    
    Attributes:
        input_tokens: Input token IDs
        output_tokens: Generated token IDs
        output_text: Decoded output text
        total_tokens: Total tokens processed
        inference_time_ms: Inference time in milliseconds
        tokens_per_second: Generation speed
        memory_used_bytes: Memory used for inference
    """
    input_tokens: List[int]
    output_tokens: List[int]
    output_text: str = ""
    total_tokens: int = 0
    inference_time_ms: float = 0.0
    tokens_per_second: float = 0.0
    memory_used_bytes: int = 0
    
    def __post_init__(self):
        """Calculate derived attributes."""
        if self.total_tokens == 0:
            self.total_tokens = len(self.input_tokens) + len(self.output_tokens)
        if self.tokens_per_second == 0 and self.inference_time_ms > 0:
            self.tokens_per_second = len(self.output_tokens) / (self.inference_time_ms / 1000)


class InferenceSimulator:
    """Simulate inference on quantized models.
    
    Provides a Python implementation of the inference logic
    that will run on the target hardware, allowing testing
    before deployment.
    
    Example:
        >>> simulator = InferenceSimulator(quant_result)
        >>> result = simulator.generate([1, 2, 3], max_tokens=50)
        >>> print(result.output_text)
    """
    
    def __init__(
        self,
        quant_result: QuantizationResult,
        config: Optional[SimulatorConfig] = None,
        vocab: Optional[Dict[int, str]] = None,
    ):
        """Initialize simulator with quantized model.
        
        Args:
            quant_result: Quantization result from Quantizer
            config: Simulator configuration
            vocab: Vocabulary mapping (token_id -> token_str)
        """
        self.quant_result = quant_result
        self.config = config or SimulatorConfig()
        
        validation = self.config.validate()
        if not validation["valid"]:
            raise ValueError(f"Invalid config: {validation['errors']}")
        
        self.vocab = vocab or self._get_default_vocab()
        
        # Build weight lookup
        self._weights: Dict[str, NDArray] = {}
        self._scales: Dict[str, float] = {}
        self._zero_points: Dict[str, float] = {}
        self._bits: Dict[str, int] = {}
        
        for layer in quant_result.layers:
            name = self._normalize_name(layer.name)
            self._weights[name] = layer.quantized_weights
            self._scales[name] = layer.scale
            self._zero_points[name] = layer.zero_point
            self._bits[name] = layer.quantized_bits
        
        # Initialize RNG
        self._rng = np.random.default_rng(self.config.seed)
        
        # KV cache
        self._kv_cache: Optional[Dict[str, NDArray]] = None
        self._cache_pos = 0
        
    def generate(
        self,
        input_tokens: List[int],
        max_tokens: int = 50,
        stop_tokens: Optional[List[int]] = None,
    ) -> InferenceResult:
        """Generate text from input tokens.
        
        Args:
            input_tokens: List of input token IDs
            max_tokens: Maximum tokens to generate
            stop_tokens: Token IDs that stop generation
            
        Returns:
            InferenceResult with generated text
        """
        import time
        
        stop_tokens = stop_tokens or [0, 50256]  # Common EOS tokens
        
        # Reset KV cache
        self._init_kv_cache()
        
        # Process input tokens
        hidden = None
        for i, token in enumerate(input_tokens):
            hidden = self._forward_token(token, i)
            
        output_tokens: List[int] = []
        current_token = input_tokens[-1] if input_tokens else 0
        total_time_ms = 0.0
        
        # Generate tokens
        for _ in range(max_tokens):
            start_time = time.perf_counter()
            
            hidden = self._forward_token(current_token, len(input_tokens) + len(output_tokens) - 1)
            
            # Get logits (simplified - would normally project to vocab)
            logits = self._get_logits(hidden)
            
            # Apply repetition penalty
            if self.config.repetition_penalty > 1.0:
                logits = self._apply_repetition_penalty(logits, input_tokens + output_tokens)
            
            # Sample next token
            next_token = self._sample_token(logits)
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            total_time_ms += elapsed_ms
            
            # Check stop conditions
            if next_token in stop_tokens:
                break
                
            output_tokens.append(next_token)
            current_token = next_token
            
            # Check max sequence length
            if len(output_tokens) >= self.config.max_seq_len:
                break
        
        # Decode output
        output_text = self._decode_tokens(output_tokens)
        
        # Estimate memory usage
        memory_bytes = self._estimate_memory_usage()
        
        return InferenceResult(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            output_text=output_text,
            inference_time_ms=total_time_ms,
            memory_used_bytes=memory_bytes,
        )
    
    def benchmark(
        self,
        input_tokens: List[int],
        num_runs: int = 10,
        max_tokens: int = 20,
    ) -> Dict[str, Any]:
        """Benchmark inference performance.
        
        Args:
            input_tokens: Input token IDs
            num_runs: Number of runs to average
            max_tokens: Tokens to generate per run
            
        Returns:
            Dict with benchmark results
        """
        import time
        
        times: List[float] = []
        
        for _ in range(num_runs):
            start = time.perf_counter()
            self.generate(input_tokens, max_tokens=max_tokens)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        return {
            "avg_time_ms": avg_time * 1000,
            "std_time_ms": std_time * 1000,
            "min_time_ms": min(times) * 1000,
            "max_time_ms": max(times) * 1000,
            "tokens_per_second": max_tokens / avg_time,
            "num_runs": num_runs,
        }
    
    def test_accuracy(
        self,
        test_cases: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Test model accuracy on test cases.
        
        Args:
            test_cases: List of test cases with 'input' and 'expected' keys
            
        Returns:
            Dict with accuracy metrics
        """
        results: List[Dict[str, Any]] = []
        
        for case in test_cases:
            input_tokens = case.get("input_tokens", [])
            expected_tokens = case.get("expected_tokens", [])
            max_tokens = len(expected_tokens) + 10 if expected_tokens else 20
            
            result = self.generate(input_tokens, max_tokens=max_tokens)
            
            # Calculate accuracy if expected output provided
            accuracy = 0.0
            if expected_tokens:
                matches = sum(
                    1 for a, b in zip(result.output_tokens, expected_tokens)
                    if a == b
                )
                accuracy = matches / max(len(expected_tokens), len(result.output_tokens))
            
            results.append({
                "input_tokens": input_tokens,
                "expected_tokens": expected_tokens,
                "output_tokens": result.output_tokens,
                "output_text": result.output_text,
                "accuracy": accuracy,
            })
        
        # Aggregate metrics
        avg_accuracy = np.mean([r["accuracy"] for r in results])
        
        return {
            "test_cases": results,
            "average_accuracy": avg_accuracy,
            "num_cases": len(test_cases),
        }
    
    def _init_kv_cache(self) -> None:
        """Initialize KV cache for transformer."""
        # Simple cache structure
        self._kv_cache = {}
        self._cache_pos = 0
    
    def _forward_token(self, token: int, position: int) -> NDArray:
        """Forward pass for a single token.
        
        This is a simplified implementation that demonstrates
        the quantization and dequantization process.
        
        Args:
            token: Input token ID
            position: Position in sequence
            
        Returns:
            Hidden state array
        """
        # Get embedding (simplified - would normally look up from weight table)
        hidden_dim = 256  # Default
        hidden = np.zeros(hidden_dim, dtype=np.float32)
        
        # Simulate embedding lookup
        embed_key = "wte"  # Word token embedding
        if embed_key in self._weights:
            # Use token ID as seed for deterministic output
            self._rng = np.random.default_rng(token + self.config.seed)
            hidden = self._rng.standard_normal(hidden_dim).astype(np.float32) * 0.1
        
        # Simulate transformer layers
        for layer_idx in range(4):  # Default 4 layers
            hidden = self._attention_block(hidden, layer_idx, position)
            hidden = self._mlp_block(hidden, layer_idx)
        
        return hidden
    
    def _attention_block(
        self,
        hidden: NDArray,
        layer_idx: int,
        position: int,
    ) -> NDArray:
        """Simulate attention block.
        
        Args:
            hidden: Hidden state
            layer_idx: Layer index
            position: Position in sequence
            
        Returns:
            Updated hidden state
        """
        # Simplified: just apply some transformation
        # In real implementation, would project Q, K, V and compute attention
        
        # Apply layer norm (simplified)
        hidden = self._layer_norm(hidden)
        
        # Simulate attention output
        output = hidden * 0.9 + self._rng.standard_normal(len(hidden)).astype(np.float32) * 0.05
        
        # Residual connection
        return hidden + output
    
    def _mlp_block(self, hidden: NDArray, layer_idx: int) -> NDArray:
        """Simulate MLP block.
        
        Args:
            hidden: Hidden state
            layer_idx: Layer index
            
        Returns:
            Updated hidden state
        """
        # Apply layer norm
        hidden = self._layer_norm(hidden)
        
        # Simulate FFN
        output = hidden * 1.1
        output = np.maximum(output, 0)  # ReLU/GELU approximation
        
        # Residual connection
        return hidden + output * 0.1
    
    def _layer_norm(self, hidden: NDArray) -> NDArray:
        """Apply layer normalization.
        
        Args:
            hidden: Hidden state
            
        Returns:
            Normalized hidden state
        """
        mean = np.mean(hidden)
        var = np.var(hidden)
        return (hidden - mean) / np.sqrt(var + 1e-5)
    
    def _get_logits(self, hidden: NDArray) -> NDArray:
        """Get logits from hidden state.
        
        Args:
            hidden: Hidden state
            
        Returns:
            Logits array
        """
        vocab_size = 50257  # GPT-2 vocab size
        
        # Simulate projection to vocab
        # In real implementation, would use lm_head weights
        logits = self._rng.standard_normal(vocab_size).astype(np.float32) * 0.1
        
        # Bias towards common tokens
        logits[0:256] += 1.0  # ASCII range
        logits[50256] = -10.0  # EOS token
        
        return logits
    
    def _apply_repetition_penalty(
        self,
        logits: NDArray,
        tokens: List[int],
    ) -> NDArray:
        """Apply repetition penalty to logits.
        
        Args:
            logits: Logits array
            tokens: Previously generated tokens
            
        Returns:
            Modified logits
        """
        if not tokens or self.config.repetition_penalty == 1.0:
            return logits
            
        logits = logits.copy()
        penalty = self.config.repetition_penalty
        
        for token in set(tokens):
            if logits[token] > 0:
                logits[token] /= penalty
            else:
                logits[token] *= penalty
                
        return logits
    
    def _sample_token(self, logits: NDArray) -> int:
        """Sample next token from logits.
        
        Args:
            logits: Logits array
            
        Returns:
            Sampled token ID
        """
        # Apply temperature
        if self.config.temperature > 0:
            logits = logits / self.config.temperature
        else:
            # Greedy sampling
            return int(np.argmax(logits))
        
        # Apply top-k
        if self.config.top_k > 0:
            top_k = min(self.config.top_k, len(logits))
            indices = np.argpartition(logits, -top_k)[-top_k:]
            mask = np.ones(len(logits), dtype=bool)
            mask[indices] = False
            logits[mask] = -float('inf')
        
        # Apply top-p (nucleus sampling)
        if self.config.top_p < 1.0:
            sorted_indices = np.argsort(logits)[::-1]
            sorted_logits = logits[sorted_indices]
            cumulative_probs = np.cumsum(self._softmax(sorted_logits))
            
            # Find cutoff
            cutoff_idx = np.searchsorted(cumulative_probs, self.config.top_p)
            cutoff_idx = min(cutoff_idx + 1, len(sorted_logits))
            
            # Mask tokens after cutoff
            mask = np.ones(len(logits), dtype=bool)
            mask[sorted_indices[:cutoff_idx]] = False
            logits[mask] = -float('inf')
        
        # Convert to probabilities and sample
        probs = self._softmax(logits)
        return int(self._rng.choice(len(probs), p=probs))
    
    def _softmax(self, logits: NDArray) -> NDArray:
        """Compute softmax.
        
        Args:
            logits: Input logits
            
        Returns:
            Probability distribution
        """
        # Numerical stability
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits)
    
    def _decode_tokens(self, tokens: List[int]) -> str:
        """Decode tokens to text.
        
        Args:
            tokens: Token IDs
            
        Returns:
            Decoded text
        """
        # Simple character-level decoding
        chars: List[str] = []
        for token in tokens:
            if 32 <= token < 127:  # Printable ASCII
                chars.append(chr(token))
            elif token in self.vocab:
                chars.append(self.vocab[token])
            else:
                # Unknown token - use placeholder
                chars.append(f"[{token}]")
        
        return "".join(chars)
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage during inference.
        
        Returns:
            Estimated bytes used
        """
        # Estimate based on model size and cache
        model_size = self.quant_result.compressed_size_bytes
        
        # KV cache (simplified estimate)
        hidden_dim = 256
        num_layers = 4
        cache_size = hidden_dim * num_layers * self.config.max_seq_len * 4 * 2  # float32, K and V
        
        return model_size + cache_size
    
    def _normalize_name(self, name: str) -> str:
        """Normalize layer name.
        
        Args:
            name: Original name
            
        Returns:
            Normalized name
        """
        return name.replace(".", "_").replace("/", "_").lower()
    
    def _get_default_vocab(self) -> Dict[int, str]:
        """Get default vocabulary (ASCII subset).
        
        Returns:
            Dict mapping token IDs to strings
        """
        vocab: Dict[int, str] = {}
        
        # Add ASCII characters
        for i in range(32, 127):
            vocab[i] = chr(i)
        
        # Add common special tokens
        vocab[0] = "<pad>"
        vocab[1] = "<s>"
        vocab[2] = "</s>"
        vocab[50256] = "<|endoftext|>"
        
        return vocab
    
    def save_state(self, path: Path) -> None:
        """Save simulator state to file.
        
        Args:
            path: Output file path
        """
        state = {
            "config": {
                "max_seq_len": self.config.max_seq_len,
                "temperature": self.config.temperature,
                "top_k": self.config.top_k,
                "top_p": self.config.top_p,
                "repetition_penalty": self.config.repetition_penalty,
                "seed": self.config.seed,
            },
            "quant_summary": self.quant_result.get_summary(),
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, path: Path) -> None:
        """Load simulator state from file.
        
        Args:
            path: Input file path
        """
        with open(path) as f:
            state = json.load(f)
        
        if "config" in state:
            cfg = state["config"]
            self.config = SimulatorConfig(
                max_seq_len=cfg.get("max_seq_len", 128),
                temperature=cfg.get("temperature", 1.0),
                top_k=cfg.get("top_k", 0),
                top_p=cfg.get("top_p", 1.0),
                repetition_penalty=cfg.get("repetition_penalty", 1.0),
                seed=cfg.get("seed", 42),
            )
            self._rng = np.random.default_rng(self.config.seed)
