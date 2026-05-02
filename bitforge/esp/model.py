"""Tiny ESP-level controller model.

This is the first step toward a real on-device assistant: a very small,
rule-friendly controller that can classify simple intents and output device
actions. It is intentionally tiny and trainable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from ..context import ContextCompressor, ContextCompressionConfig
from ..prune import BlockPruner, PruningConfig
from ..kvcache import KVCacheCompressor, KVCacheConfig


@dataclass
class EspTaskExample:
    text: str
    label: str


@dataclass
class EspTaskPrediction:
    label: str
    confidence: float
    actions: List[str]
    tokens_per_second_estimate: float


class TinyEspController:
    """A tiny rule+score controller intended for microcontroller demos.

    It does not pretend to be a general chatbot. It predicts one of a small
    number of device-level intents and emits compact actions.
    """

    def __init__(self):
        self.intent_labels = [
            "control_light",
            "set_mode",
            "status_check",
            "help",
            "idle",
        ]
        self.context = ContextCompressor(ContextCompressionConfig(max_tokens=48, keep_first=6, keep_recent=12, keep_every=4))
        self.pruner = BlockPruner(PruningConfig(block_size=16, target_keep_ratio=0.25))
        self.kv = KVCacheCompressor(KVCacheConfig(recent_bits=4, mid_bits=3, old_bits=2))
        self.weights = self._build_default_weights()

    def _build_default_weights(self) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(42)
        return {
            "intent": rng.standard_normal((len(self.intent_labels), 32)).astype(np.float32) * 0.05,
            "action": rng.standard_normal((8, 32)).astype(np.float32) * 0.03,
        }

    def fit(self, examples: Sequence[EspTaskExample]) -> None:
        # Lightweight “training”: count label words into the prototype weights.
        counts = {label: np.zeros(32, dtype=np.float32) for label in self.intent_labels}
        for ex in examples:
            vec = self._embed(ex.text)
            counts.setdefault(ex.label, np.zeros(32, dtype=np.float32))
            counts[ex.label] += vec
        for i, label in enumerate(self.intent_labels):
            self.weights["intent"][i] = counts[label] / max(1.0, np.linalg.norm(counts[label]) + 1e-6)

    def predict(self, text: str) -> EspTaskPrediction:
        tokens = self._tokenize(text)
        compact = self.context.compress_tokens(tokens)
        vec = self._embed(" ".join(str(t) for t in compact.compressed_tokens))
        scores = self.weights["intent"] @ vec
        idx = int(np.argmax(scores))
        label = self.intent_labels[idx]
        confidence = float(self._softmax(scores)[idx])
        actions = self._actions_for_label(label, text)
        tps = self._estimate_tokens_per_second(len(compact.compressed_tokens), label)
        return EspTaskPrediction(label=label, confidence=confidence, actions=actions, tokens_per_second_estimate=tps)

    def export_tiny_profile(self) -> Dict[str, object]:
        pruned = self.pruner.prune_weights({"intent": self.weights["intent"], "action": self.weights["action"]})
        return {
            "intent_labels": self.intent_labels,
            "pruned_bytes": pruned.sparse_nbytes,
            "original_bytes": pruned.original_nbytes,
            "compression_ratio": pruned.compression_ratio,
            "context_window": self.context.config.max_tokens,
            "kv_bits": self.kv.config.recent_bits,
        }

    def _actions_for_label(self, label: str, text: str) -> List[str]:
        if label == "control_light":
            return ["gpio:set=1" if any(w in text.lower() for w in ["on", "bright", "open"]) else "gpio:set=0"]
        if label == "set_mode":
            return ["mode:performance" if "fast" in text.lower() else "mode:eco"]
        if label == "status_check":
            return ["status:read"]
        if label == "help":
            return ["help:show"]
        return ["idle:keep"]

    def _tokenize(self, text: str) -> List[int]:
        return [min(255, ord(c)) for c in text[:128]]

    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros(32, dtype=np.float32)
        for i, ch in enumerate(text.lower()):
            vec[i % 32] += (ord(ch) % 37) / 37.0
        norm = np.linalg.norm(vec) + 1e-6
        return vec / norm

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        z = logits - np.max(logits)
        e = np.exp(z)
        return e / np.sum(e)

    def _estimate_tokens_per_second(self, token_count: int, label: str) -> float:
        base = 250.0 if label != "idle" else 400.0
        return max(1.0, base - token_count * 1.5)
