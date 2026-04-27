import json
import math
from pathlib import Path
from typing import Dict, List, Optional


FEATURE_ORDER = ["is_major", "tempo_norm", "energy"]


class LinearPreferenceModel:
    def __init__(self, weights: Optional[List[float]] = None, bias: float = 0.0):
        self.weights = weights or [0.0 for _ in FEATURE_ORDER]
        self.bias = bias

    def featurize(self, candidate: Dict) -> List[float]:
        return [float(candidate["features"][name]) for name in FEATURE_ORDER]

    def score_features(self, features: List[float]) -> float:
        return sum(weight * value for weight, value in zip(self.weights, features)) + self.bias

    def score(self, candidate: Dict) -> float:
        return self.score_features(self.featurize(candidate))

    def preference_probability(self, preferred: Dict, other: Dict) -> float:
        margin = self.score(preferred) - self.score(other)
        if margin >= 0:
            return 1.0 / (1.0 + math.exp(-margin))
        exp_margin = math.exp(margin)
        return exp_margin / (1.0 + exp_margin)

    def train(self, pairs: List[Dict], learning_rate: float, epochs: int) -> List[Dict]:
        history = []
        for epoch in range(epochs):
            total_loss = 0.0
            for pair in pairs:
                preferred = self.featurize(pair["preferred"])
                other = self.featurize(pair["other"])
                diff = [pref - oth for pref, oth in zip(preferred, other)]
                margin = sum(weight * value for weight, value in zip(self.weights, diff))
                loss = math.log1p(math.exp(-margin))
                scale = -1.0 / (1.0 + math.exp(margin))
                for idx, value in enumerate(diff):
                    self.weights[idx] -= learning_rate * scale * value
                total_loss += loss
            history.append(
                {
                    "epoch": epoch + 1,
                    "avg_loss": total_loss / max(1, len(pairs)),
                    "weights": list(self.weights),
                    "bias": self.bias,
                }
            )
        return history

    def save(self, path: Path) -> None:
        payload = {
            "feature_order": FEATURE_ORDER,
            "weights": self.weights,
            "bias": self.bias,
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    @classmethod
    def load(cls, path: Path) -> "LinearPreferenceModel":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(weights=payload["weights"], bias=payload["bias"])
