"""Tiny ESP-level controller model.

This controller is intentionally small, trainable, and more capable than a
pure keyword toy. It is designed for constrained devices where the job is to
turn short natural-language commands into a few deterministic actions.

The trick is not pretending to be a huge chatbot. The trick is having a tiny
policy model that can learn several task families and stay fast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import math
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
    explanation: str = ""


@dataclass
class EspTrainingReport:
    examples_seen: int
    labels_seen: List[str]
    accuracy_on_train: float
    avg_confidence: float
    profile_bytes: int


class TinyEspController:
    """A tiny controller for ESP-style task routing.

    It learns a compact prototype for each intent. At inference time it uses
    hashed bag-of-words features, a small amount of rule bias, and a few slot
    extraction helpers to emit device actions.
    """

    DEFAULT_LABELS = [
        "control_light",
        "set_mode",
        "status_check",
        "help",
        "idle",
        "set_timer",
        "launch_script",
        "play_media",
        "read_sensor",
        "connect_wifi",
        "reboot_device",
        "set_temperature",
    ]

    def __init__(self, feature_dim: int = 96):
        self.feature_dim = feature_dim
        self.intent_labels = list(self.DEFAULT_LABELS)
        self.context = ContextCompressor(
            ContextCompressionConfig(
                max_tokens=48,
                keep_first=6,
                keep_recent=12,
                keep_every=4,
            )
        )
        self.pruner = BlockPruner(PruningConfig(block_size=16, target_keep_ratio=0.25))
        self.kv = KVCacheCompressor(KVCacheConfig(recent_bits=4, mid_bits=3, old_bits=2))
        self.label_counts: Dict[str, int] = {label: 0 for label in self.intent_labels}
        self.label_vectors: Dict[str, np.ndarray] = {
            label: np.zeros(self.feature_dim, dtype=np.float32) for label in self.intent_labels
        }
        self.keyword_bias: Dict[str, Dict[str, float]] = {label: {} for label in self.intent_labels}
        self.trained_examples: List[EspTaskExample] = []
        self._trained = False

    @property
    def is_trained(self) -> bool:
        return self._trained

    def fit(self, examples: Sequence[EspTaskExample]) -> EspTrainingReport:
        """Train a compact prototype model from examples."""
        if not examples:
            raise ValueError("TinyEspController.fit requires at least one example")

        self._ensure_labels(examples)
        self.label_counts = {label: 0 for label in self.intent_labels}
        self.label_vectors = {
            label: np.zeros(self.feature_dim, dtype=np.float32) for label in self.intent_labels
        }
        self.keyword_bias = {label: {} for label in self.intent_labels}
        self.trained_examples = list(examples)

        token_totals: Dict[str, Dict[str, int]] = {label: {} for label in self.intent_labels}
        label_examples: Dict[str, List[EspTaskExample]] = {label: [] for label in self.intent_labels}

        for ex in examples:
            label_examples.setdefault(ex.label, []).append(ex)
            self.label_counts[ex.label] = self.label_counts.get(ex.label, 0) + 1
            vec = self._embed(ex.text)
            self.label_vectors[ex.label] = self.label_vectors.get(ex.label, np.zeros(self.feature_dim, dtype=np.float32)) + vec
            for token in self._tokenize_words(ex.text):
                token_totals[ex.label][token] = token_totals[ex.label].get(token, 0) + 1

        for label, total in self.label_vectors.items():
            count = max(1, self.label_counts.get(label, 0))
            self.label_vectors[label] = total / float(count)
            norm = np.linalg.norm(self.label_vectors[label]) + 1e-6
            self.label_vectors[label] = self.label_vectors[label] / norm

        for label, counts in token_totals.items():
            for token, count in counts.items():
                self.keyword_bias[label][token] = math.log1p(count)

        predictions = [self.predict(ex.text).label for ex in examples]
        correct = sum(1 for pred, ex in zip(predictions, examples) if pred == ex.label)
        confidences = [self.predict(ex.text).confidence for ex in examples]
        profile = self.export_tiny_profile()
        self._trained = True

        return EspTrainingReport(
            examples_seen=len(examples),
            labels_seen=sorted(self.label_counts.keys()),
            accuracy_on_train=correct / len(examples),
            avg_confidence=float(np.mean(confidences)) if confidences else 0.0,
            profile_bytes=int(profile["estimated_profile_bytes"]),
        )

    def predict(self, text: str) -> EspTaskPrediction:
        """Predict a compact action intent for the given text."""
        tokens = self._tokenize(text)
        compact = self.context.compress_tokens(tokens)
        compact_text = self._tokens_to_text(compact.compressed_tokens)
        feature_text = text + " " + compact_text
        vector = self._embed(feature_text)

        scores = self._score_labels(feature_text, vector)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        label, raw_score = ranked[0]
        confidence = self._softmax_from_scores(scores)[label]
        actions = self._actions_for_label(label, text)
        tps = self._estimate_tokens_per_second(len(compact.compressed_tokens), label)
        explanation = self._explain_choice(label, ranked)

        return EspTaskPrediction(
            label=label,
            confidence=float(confidence),
            actions=actions,
            tokens_per_second_estimate=tps,
            explanation=explanation,
        )

    def batch_predict(self, texts: Sequence[str]) -> List[EspTaskPrediction]:
        return [self.predict(text) for text in texts]

    def evaluate(self, examples: Sequence[EspTaskExample]) -> Dict[str, float]:
        if not examples:
            return {"accuracy": 0.0, "avg_confidence": 0.0, "count": 0.0}
        preds = [self.predict(ex.text) for ex in examples]
        accuracy = sum(1 for pred, ex in zip(preds, examples) if pred.label == ex.label) / len(examples)
        avg_confidence = float(np.mean([pred.confidence for pred in preds]))
        return {"accuracy": float(accuracy), "avg_confidence": avg_confidence, "count": float(len(examples))}

    def export_tiny_profile(self) -> Dict[str, object]:
        pruned = self.pruner.prune_weights(
            {
                "intent": self.label_vectors_to_matrix(),
                "bias": self.keyword_bias_to_matrix(),
            }
        )
        estimated_profile_bytes = self.feature_dim * len(self.intent_labels) * 4 + pruned.sparse_nbytes
        return {
            "intent_labels": self.intent_labels,
            "examples_seen": len(self.trained_examples),
            "trained": self._trained,
            "profile_original_bytes": pruned.original_nbytes,
            "profile_pruned_bytes": pruned.sparse_nbytes,
            "compression_ratio": pruned.compression_ratio,
            "context_window": self.context.config.max_tokens,
            "kv_bits": self.kv.config.recent_bits,
            "estimated_profile_bytes": int(estimated_profile_bytes),
            "feature_dim": self.feature_dim,
        }

    def save_profile(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.export_tiny_profile(), indent=2))

    @classmethod
    def load_profile(cls, path: Path) -> "TinyEspController":
        data = json.loads(path.read_text())
        controller = cls(feature_dim=int(data.get("feature_dim", 96)))
        controller.intent_labels = list(data.get("intent_labels", controller.intent_labels))
        controller.label_counts = {label: 1 for label in controller.intent_labels}
        controller._trained = bool(data.get("trained", False))
        return controller

    def default_training_set(self) -> List[EspTaskExample]:
        return [
            EspTaskExample("turn on the light", "control_light"),
            EspTaskExample("switch the lamp off", "control_light"),
            EspTaskExample("set performance mode", "set_mode"),
            EspTaskExample("switch to eco mode", "set_mode"),
            EspTaskExample("what is the status", "status_check"),
            EspTaskExample("check system status", "status_check"),
            EspTaskExample("help me", "help"),
            EspTaskExample("what can you do", "help"),
            EspTaskExample("set a timer for 10 minutes", "set_timer"),
            EspTaskExample("remind me in 5 minutes", "set_timer"),
            EspTaskExample("launch diagnostics script", "launch_script"),
            EspTaskExample("run the maintenance script", "launch_script"),
            EspTaskExample("play music", "play_media"),
            EspTaskExample("start playback", "play_media"),
            EspTaskExample("read the temperature sensor", "read_sensor"),
            EspTaskExample("check sensor value", "read_sensor"),
            EspTaskExample("connect to wifi", "connect_wifi"),
            EspTaskExample("join the network", "connect_wifi"),
            EspTaskExample("reboot device", "reboot_device"),
            EspTaskExample("restart now", "reboot_device"),
            EspTaskExample("set temperature to 21 degrees", "set_temperature"),
            EspTaskExample("make it 18 celsius", "set_temperature"),
        ]

    def train_from_defaults(self) -> EspTrainingReport:
        return self.fit(self.default_training_set())

    def _ensure_labels(self, examples: Sequence[EspTaskExample]) -> None:
        labels = set(self.intent_labels)
        for ex in examples:
            labels.add(ex.label)
        self.intent_labels = sorted(labels)
        for label in self.intent_labels:
            self.label_counts.setdefault(label, 0)
            self.label_vectors.setdefault(label, np.zeros(self.feature_dim, dtype=np.float32))
            self.keyword_bias.setdefault(label, {})

    def _tokenize_words(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9']+", text.lower())

    def _tokenize(self, text: str) -> List[int]:
        return [min(255, ord(c)) for c in text[:160]]

    def _tokens_to_text(self, tokens: Sequence[int]) -> str:
        return "".join(chr(t) for t in tokens if 32 <= t < 127)

    def _stable_hash(self, word: str) -> int:
        digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little", signed=False)

    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.feature_dim, dtype=np.float32)
        words = self._tokenize_words(text)
        for i, word in enumerate(words):
            bucket = self._stable_hash(word) % self.feature_dim
            vec[bucket] += 1.0 + (len(word) / 12.0)
            vec[(bucket + i) % self.feature_dim] += 0.15
        for i, ch in enumerate(text.lower()):
            vec[(ord(ch) + i) % self.feature_dim] += 0.02
        norm = np.linalg.norm(vec) + 1e-6
        return vec / norm

    def _score_labels(self, text: str, vector: np.ndarray) -> Dict[str, float]:
        words = self._tokenize_words(text)
        scores: Dict[str, float] = {}
        for label in self.intent_labels:
            proto = self.label_vectors.get(label)
            if proto is None or not proto.any():
                proto = np.zeros(self.feature_dim, dtype=np.float32)
            cos = float(np.dot(vector, proto))
            keyword_score = sum(self.keyword_bias.get(label, {}).get(word, 0.0) for word in words)
            prior = math.log1p(self.label_counts.get(label, 0))
            rule_boost = self._rule_boost(label, words)
            scores[label] = cos * 2.4 + 0.18 * keyword_score + 0.12 * prior + rule_boost
        return scores

    def _softmax_from_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        labels = list(scores.keys())
        values = np.array([scores[label] for label in labels], dtype=np.float32)
        values = values - np.max(values)
        exp = np.exp(values)
        probs = exp / np.sum(exp)
        return {label: float(prob) for label, prob in zip(labels, probs)}

    def _rule_boost(self, label: str, words: Sequence[str]) -> float:
        text = " ".join(words)
        if label == "control_light" and any(w in text for w in ["light", "lamp", "led", "on", "off"]):
            return 1.2
        if label == "set_mode" and any(w in text for w in ["mode", "eco", "performance", "fast", "safe"]):
            return 1.1
        if label == "status_check" and any(w in text for w in ["status", "state", "health", "report"]):
            return 1.0
        if label == "help" and any(w in text for w in ["help", "what", "can you do", "commands"]):
            return 0.9
        if label == "set_timer" and any(w in text for w in ["timer", "minutes", "minute", "remind"]):
            return 1.1
        if label == "launch_script" and any(w in text for w in ["script", "run", "launch", "diagnostics"]):
            return 1.0
        if label == "play_media" and any(w in text for w in ["play", "music", "audio", "song", "media"]):
            return 1.0
        if label == "read_sensor" and any(w in text for w in ["sensor", "temperature", "value", "read"]):
            return 1.1
        if label == "connect_wifi" and any(w in text for w in ["wifi", "network", "connect", "join"]):
            return 1.2
        if label == "reboot_device" and any(w in text for w in ["reboot", "restart", "reset"]):
            return 1.2
        if label == "set_temperature" and any(w in text for w in ["temperature", "degrees", "celsius", "temp"]):
            return 1.2
        return 0.0

    def _actions_for_label(self, label: str, text: str) -> List[str]:
        lower = text.lower()
        if label == "control_light":
            return ["gpio:set=1" if any(w in lower for w in ["on", "bright", "open", "enable"]) else "gpio:set=0"]
        if label == "set_mode":
            if any(w in lower for w in ["fast", "performance"]):
                return ["mode:performance"]
            if any(w in lower for w in ["eco", "safe"]):
                return ["mode:eco"]
            return ["mode:auto"]
        if label == "status_check":
            return ["status:read", "status:report"]
        if label == "help":
            return ["help:show"]
        if label == "idle":
            return ["idle:keep"]
        if label == "set_timer":
            minutes = self._extract_number(lower) or 5
            return [f"timer:start={minutes}m"]
        if label == "launch_script":
            script = self._extract_word_after(lower, ["script", "run", "launch"]) or "default"
            return [f"script:run={script}"]
        if label == "play_media":
            return ["media:play"]
        if label == "read_sensor":
            sensor = self._extract_sensor(lower)
            return [f"sensor:read={sensor}"]
        if label == "connect_wifi":
            return ["network:connect"]
        if label == "reboot_device":
            return ["system:reboot"]
        if label == "set_temperature":
            temp = self._extract_temperature(lower) or 21
            return [f"thermostat:set={temp}c"]
        return ["idle:keep"]

    def _extract_number(self, text: str) -> Optional[int]:
        match = re.search(r"(\d+)", text)
        return int(match.group(1)) if match else None

    def _extract_word_after(self, text: str, anchors: Sequence[str]) -> Optional[str]:
        for anchor in anchors:
            m = re.search(rf"{re.escape(anchor)}\s+([a-z0-9_\-]+)", text)
            if m:
                return m.group(1)
        return None

    def _extract_sensor(self, text: str) -> str:
        for sensor in ["temperature", "humidity", "pressure", "light", "motion"]:
            if sensor in text:
                return sensor
        return "default"

    def _extract_temperature(self, text: str) -> Optional[int]:
        m = re.search(r"(\d+)\s*(?:c|celsius|degrees?)", text)
        return int(m.group(1)) if m else None

    def _estimate_tokens_per_second(self, token_count: int, label: str) -> float:
        base = 420.0 if label in {"idle", "help", "status_check"} else 360.0
        penalty = min(250.0, token_count * 2.5)
        return max(30.0, base - penalty)

    def _explain_choice(self, label: str, ranked: Sequence[Tuple[str, float]]) -> str:
        top3 = ", ".join(f"{name}:{score:.2f}" for name, score in ranked[:3])
        return f"picked {label} from {top3}"

    def label_vectors_to_matrix(self) -> np.ndarray:
        rows = [self.label_vectors.get(label, np.zeros(self.feature_dim, dtype=np.float32)) for label in self.intent_labels]
        return np.stack(rows).astype(np.float32)

    def keyword_bias_to_matrix(self) -> np.ndarray:
        return np.zeros((len(self.intent_labels), self.feature_dim), dtype=np.float32)
