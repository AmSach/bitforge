"""
Model loading utilities.

Supports loading models from Hugging Face Hub, local safetensors,
and other common formats.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import logging
import os

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model architecture configuration.
    
    Attributes:
        hidden_size: Hidden layer dimension
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        intermediate_size: FFN intermediate dimension
        vocab_size: Vocabulary size
        max_position_embeddings: Maximum sequence length
        layer_norm_eps: Layer normalization epsilon
        type_vocab_size: Token type vocabulary size
        activation_function: Activation function name
        model_type: Model architecture type
    """
    hidden_size: int = 256
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    intermediate_size: int = 1024
    vocab_size: int = 50257
    max_position_embeddings: int = 128
    layer_norm_eps: float = 1e-5
    type_vocab_size: int = 1
    activation_function: str = "gelu"
    model_type: str = "gpt2"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create config from dictionary.
        
        Args:
            data: Configuration dictionary
            
        Returns:
            ModelConfig instance
        """
        # Handle common naming variations
        mapping = {
            "hidden_dim": "hidden_size",
            "n_embd": "hidden_size",
            "d_model": "hidden_size",
            "n_layer": "num_hidden_layers",
            "num_layers": "num_hidden_layers",
            "n_head": "num_attention_heads",
            "num_heads": "num_attention_heads",
            "n_positions": "max_position_embeddings",
            "max_seq_len": "max_position_embeddings",
        }
        
        normalized = {}
        for key, value in data.items():
            normalized_key = mapping.get(key.lower(), key.lower())
            if normalized_key not in normalized:
                normalized[normalized_key] = value
                
        return cls(
            hidden_size=normalized.get("hidden_size", 256),
            num_hidden_layers=normalized.get("num_hidden_layers", 4),
            num_attention_heads=normalized.get("num_attention_heads", 4),
            intermediate_size=normalized.get("intermediate_size", 1024),
            vocab_size=normalized.get("vocab_size", 50257),
            max_position_embeddings=normalized.get("max_position_embeddings", 128),
            layer_norm_eps=normalized.get("layer_norm_eps", 1e-5),
            type_vocab_size=normalized.get("type_vocab_size", 1),
            activation_function=normalized.get("activation_function", "gelu"),
            model_type=normalized.get("model_type", "gpt2"),
        )


@dataclass
class LoadedModel:
    """A loaded model ready for compression.
    
    Attributes:
        name: Model name or identifier
        config: Model architecture configuration
        weights: Dictionary of weight tensors
        param_count: Total parameter count
        size_bytes: Total size in bytes
        dtype: Data type of weights
        tokenizer_name: Tokenizer identifier (if available)
    """
    name: str
    config: ModelConfig
    weights: Dict[str, NDArray[np.floating]]
    param_count: int = 0
    size_bytes: int = 0
    dtype: str = "float32"
    tokenizer_name: Optional[str] = None
    
    def __post_init__(self):
        """Calculate derived attributes."""
        if self.param_count == 0:
            self.param_count = sum(
                int(np.prod(w.shape)) for w in self.weights.values()
            )
        if self.size_bytes == 0:
            dtype_size = 4 if "float32" in self.dtype else 2
            self.size_bytes = self.param_count * dtype_size


class ModelLoader:
    """Load models from various sources.
    
    Supports:
    - Hugging Face Hub (model_id)
    - Local safetensors files
    - Local PyTorch checkpoint files
    
    Example:
        >>> loader = ModelLoader()
        >>> model = loader.load("gpt2")
        >>> model = loader.load("./my_model.safetensors")
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize model loader.
        
        Args:
            cache_dir: Directory to cache downloaded models
        """
        self.cache_dir = cache_dir or Path.home() / ".cache" / "bitforge"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def load(
        self,
        source: str,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> LoadedModel:
        """Load a model from source.
        
        Args:
            source: Model ID, path to safetensors, or path to checkpoint
            revision: Optional model revision (for HuggingFace)
            trust_remote_code: Whether to trust remote code
            
        Returns:
            LoadedModel instance
            
        Raises:
            FileNotFoundError: If local file not found
            ValueError: If source cannot be loaded
        """
        # Check if source is a local file
        if os.path.exists(source):
            if source.endswith(".safetensors"):
                return self._load_safetensors(source)
            elif source.endswith((".bin", ".pt", ".pth")):
                return self._load_pytorch(source)
            elif os.path.isdir(source):
                return self._load_directory(source)
            else:
                raise ValueError(f"Unknown file format: {source}")
        
        # Otherwise, treat as HuggingFace model ID
        return self._load_from_hf(source, revision, trust_remote_code)
    
    def _load_safetensors(self, path: str) -> LoadedModel:
        """Load model from safetensors file.
        
        Args:
            path: Path to safetensors file
            
        Returns:
            LoadedModel instance
        """
        try:
            from safetensors import safe_open
        except ImportError:
            raise ImportError("safetensors is required. Install with: pip install safetensors")
            
        weights: Dict[str, NDArray[np.floating]] = {}
        dtype = "float32"
        
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                weights[key] = tensor.numpy()
                if dtype == "float32" and tensor.dtype == torch.float16:
                    dtype = "float16"
        
        # Try to load config from same directory
        model_dir = Path(path).parent
        config = self._load_config_from_dir(model_dir)
        
        model_name = Path(path).stem
        
        return LoadedModel(
            name=model_name,
            config=config,
            weights=weights,
            dtype=dtype,
        )
    
    def _load_pytorch(self, path: str) -> LoadedModel:
        """Load model from PyTorch checkpoint.
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            LoadedModel instance
        """
        checkpoint = torch.load(path, map_location="cpu")
        
        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
            
        weights: Dict[str, NDArray[np.floating]] = {}
        dtype = "float32"
        
        for key, tensor in state_dict.items():
            if isinstance(tensor, Tensor):
                weights[key] = tensor.detach().numpy()
                if dtype == "float32" and tensor.dtype == torch.float16:
                    dtype = "float16"
        
        # Try to load config
        model_dir = Path(path).parent
        config = self._load_config_from_dir(model_dir)
        
        model_name = Path(path).stem
        
        return LoadedModel(
            name=model_name,
            config=config,
            weights=weights,
            dtype=dtype,
        )
    
    def _load_directory(self, directory: str) -> LoadedModel:
        """Load model from directory containing model files.
        
        Args:
            directory: Path to model directory
            
        Returns:
            LoadedModel instance
        """
        dir_path = Path(directory)
        
        # Find model file
        safetensors_files = list(dir_path.glob("*.safetensors"))
        bin_files = list(dir_path.glob("*.bin"))
        
        if safetensors_files:
            return self._load_safetensors(str(safetensors_files[0]))
        elif bin_files:
            return self._load_pytorch(str(bin_files[0]))
        else:
            raise FileNotFoundError(f"No model files found in {directory}")
    
    def _load_from_hf(
        self,
        model_id: str,
        revision: Optional[str],
        trust_remote_code: bool,
    ) -> LoadedModel:
        """Load model from Hugging Face Hub.
        
        Args:
            model_id: HuggingFace model identifier
            revision: Optional revision
            trust_remote_code: Whether to trust remote code
            
        Returns:
            LoadedModel instance
        """
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
            from transformers import AutoConfig, AutoModelForCausalLM
        except ImportError:
            raise ImportError(
                "huggingface_hub and transformers are required. "
                "Install with: pip install huggingface_hub transformers"
            )
        
        logger.info(f"Loading model from HuggingFace: {model_id}")
        
        # Download model
        local_dir = snapshot_download(
            model_id,
            revision=revision,
            cache_dir=str(self.cache_dir),
        )
        
        # Load config
        try:
            config = AutoConfig.from_pretrained(
                model_id,
                revision=revision,
                trust_remote_code=trust_remote_code,
            )
            model_config = ModelConfig.from_dict(config.to_dict())
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
            model_config = ModelConfig()
        
        # Load model weights
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                revision=revision,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.float32,
            )
            state_dict = model.state_dict()
        except Exception as e:
            logger.warning(f"Could not load model class, loading weights directly: {e}")
            # Fallback to loading safetensors directly
            safetensors_path = hf_hub_download(
                model_id,
                filename="model.safetensors",
                revision=revision,
                cache_dir=str(self.cache_dir),
            )
            return self._load_safetensors(safetensors_path)
        
        # Convert to numpy
        weights: Dict[str, NDArray[np.floating]] = {}
        dtype = "float32"
        
        for key, tensor in state_dict.items():
            weights[key] = tensor.detach().numpy()
            if dtype == "float32" and tensor.dtype == torch.float16:
                dtype = "float16"
        
        return LoadedModel(
            name=model_id,
            config=model_config,
            weights=weights,
            dtype=dtype,
            tokenizer_name=model_id,
        )
    
    def _load_config_from_dir(self, directory: Path) -> ModelConfig:
        """Try to load config from directory.
        
        Args:
            directory: Directory to search
            
        Returns:
            ModelConfig (defaults if not found)
        """
        config_files = [
            directory / "config.json",
            directory / "model_config.json",
        ]
        
        for config_file in config_files:
            if config_file.exists():
                try:
                    with open(config_file) as f:
                        config_dict = json.load(f)
                    return ModelConfig.from_dict(config_dict)
                except Exception as e:
                    logger.warning(f"Could not parse {config_file}: {e}")
        
        return ModelConfig()
    
    def list_supported_models(self) -> List[str]:
        """List some commonly supported small models.
        
        Returns:
            List of model identifiers
        """
        return [
            "gpt2",
            "gpt2-medium",
            "distilgpt2",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "Qwen/Qwen2.5-0.5B",
            "Qwen/Qwen2.5-1.5B",
            "microsoft/Phi-3-mini-4k-instruct",
            "HuggingFaceTB/SmolLM-135M",
            "HuggingFaceTB/SmolLM-360M",
        ]
