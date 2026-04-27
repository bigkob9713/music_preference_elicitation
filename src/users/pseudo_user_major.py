import random
import math
from typing import Dict, Optional


class MajorPrefPseudoUser:
    """Deterministic pseudo-user that prefers major over minor."""

    def prefer(self, left: Dict, right: Dict) -> int:
        left_major = int(left["features"]["is_major"])
        right_major = int(right["features"]["is_major"])
        if left_major == right_major:
            return 0
        return 1 if left_major > right_major else -1


class StochasticMajorPrefPseudoUser:
    """Utility-based pseudo-user with Bradley-Terry pairwise choices."""

    def __init__(
        self,
        major_utility: float = 2.1972245773362196,
        minor_utility: float = 0.0,
        tempo_weight: float = 0.0,
        rng: Optional[random.Random] = None,
    ):
        self.major_utility = major_utility
        self.minor_utility = minor_utility
        self.tempo_weight = tempo_weight
        self.rng = rng or random.Random()

    def utility(self, candidate: Dict) -> float:
        mode_utility = self.major_utility if int(candidate["features"]["is_major"]) else self.minor_utility
        return mode_utility + self.tempo_weight * float(candidate["features"]["tempo_norm"])

    def preference_probability(self, left: Dict, right: Dict) -> float:
        utility_gap = self.utility(left) - self.utility(right)
        return 1.0 / (1.0 + math.exp(-utility_gap))

    def prefer(self, left: Dict, right: Dict) -> int:
        return 1 if self.rng.random() < self.preference_probability(left, right) else -1


def build_major_pseudo_user(config: Dict, rng: random.Random):
    user_config = config.get("user", {})
    user_type = user_config.get("type", "deterministic")

    if user_type == "deterministic":
        return MajorPrefPseudoUser()
    if user_type == "stochastic":
        return StochasticMajorPrefPseudoUser(
            major_utility=float(user_config.get("major_utility", 2.1972245773362196)),
            minor_utility=float(user_config.get("minor_utility", 0.0)),
            tempo_weight=float(user_config.get("tempo_weight", 0.0)),
            rng=rng,
        )

    raise ValueError(f"Unknown user type: {user_type}")
