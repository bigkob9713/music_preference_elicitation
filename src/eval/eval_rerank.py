import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from src.common import load_config, read_jsonl, user_metadata
from src.models.preference_model import LinearPreferenceModel


def candidate_is_major(candidate: Dict) -> int:
    return int(candidate["features"]["is_major"])


def candidate_utility(candidate: Dict, config: Dict) -> float:
    user_config = config.get("user", {})
    if user_config.get("type", "deterministic") == "stochastic":
        major_utility = float(user_config.get("major_utility", 2.1972245773362196))
        minor_utility = float(user_config.get("minor_utility", 0.0))
        tempo_weight = float(user_config.get("tempo_weight", 0.0))
        mode_utility = major_utility if candidate_is_major(candidate) else minor_utility
        return mode_utility + tempo_weight * float(candidate["features"]["tempo_norm"])
    return float(candidate_is_major(candidate))


def rerank_metrics(model: LinearPreferenceModel, rerank_sets: List[Dict], config: Dict) -> Dict:
    learned_top_major = 0
    original_top_major = 0
    random_top_major = 0
    oracle_top_major = 0
    original_top_utility = 0.0
    random_top_utility = 0.0
    learned_top_utility = 0.0
    oracle_top_utility = 0.0
    learned_major_ranks = []
    rng = random.Random(config["seed"])

    for item in rerank_sets:
        candidates = item["candidates"]
        random_top = rng.choice(candidates)

        original_top_major += candidate_is_major(candidates[0])
        random_top_major += candidate_is_major(random_top)
        original_top_utility += candidate_utility(candidates[0], config)
        random_top_utility += candidate_utility(random_top, config)

        learned_sorted = sorted(candidates, key=model.score, reverse=True)
        oracle_sorted = sorted(candidates, key=lambda candidate: candidate_utility(candidate, config), reverse=True)

        learned_top_major += candidate_is_major(learned_sorted[0])
        oracle_top_major += candidate_is_major(oracle_sorted[0])
        learned_top_utility += candidate_utility(learned_sorted[0], config)
        oracle_top_utility += candidate_utility(oracle_sorted[0], config)

        major_positions = [
            idx + 1 for idx, candidate in enumerate(learned_sorted) if candidate_is_major(candidate)
        ]
        learned_major_ranks.append(sum(major_positions) / max(1, len(major_positions)))

    total_sets = max(1, len(rerank_sets))
    return {
        "num_sets": len(rerank_sets),
        "top1_expected_utility_before_rerank": original_top_utility / total_sets,
        "top1_expected_utility_random": random_top_utility / total_sets,
        "top1_expected_utility_learned": learned_top_utility / total_sets,
        "top1_expected_utility_oracle": oracle_top_utility / total_sets,
        "top1_expected_utility_regret_learned": (oracle_top_utility - learned_top_utility) / total_sets,
        "secondary_top1_major_rate_before_rerank": original_top_major / total_sets,
        "secondary_top1_major_rate_random": random_top_major / total_sets,
        "secondary_top1_major_rate_learned": learned_top_major / total_sets,
        "secondary_top1_major_rate_oracle": oracle_top_major / total_sets,
        "secondary_rerank_improvement": (learned_top_major - original_top_major) / total_sets,
        "mean_rank_of_major_candidates_learned": sum(learned_major_ranks) / max(1, len(learned_major_ranks)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    artifact_dir = Path(config["artifact_dir"])
    rerank_sets = read_jsonl(artifact_dir / "rerank_sets.jsonl")
    model = LinearPreferenceModel.load(artifact_dir / "preference_model.json")

    metrics = rerank_metrics(model, rerank_sets, config)
    metrics.update(user_metadata(config))
    with (artifact_dir / "rerank_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
