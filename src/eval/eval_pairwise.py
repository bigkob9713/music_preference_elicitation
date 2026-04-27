import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

from src.common import load_config, read_jsonl, user_metadata
from src.models.preference_model import LinearPreferenceModel


def evaluate_pairwise_accuracy(model: LinearPreferenceModel, pairs: List[Dict]) -> Dict:
    correct = 0
    total_log_loss = 0.0
    total_brier = 0.0
    margins = []
    for pair in pairs:
        preferred_score = model.score(pair["preferred"])
        other_score = model.score(pair["other"])
        predicted_prob = model.preference_probability(pair["preferred"], pair["other"])
        if preferred_score > other_score:
            correct += 1
        total_log_loss += -math.log(max(predicted_prob, 1e-12))
        total_brier += (1.0 - predicted_prob) ** 2
        margins.append(preferred_score - other_score)
    return {
        "num_pairs": len(pairs),
        "pairwise_accuracy": correct / max(1, len(pairs)),
        "pairwise_log_loss": total_log_loss / max(1, len(pairs)),
        "pairwise_brier_score": total_brier / max(1, len(pairs)),
        "mean_margin": sum(margins) / max(1, len(margins)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    artifact_dir = Path(config["artifact_dir"])
    pairs = read_jsonl(artifact_dir / "test_pairs.jsonl")
    model = LinearPreferenceModel.load(artifact_dir / "preference_model.json")

    metrics = evaluate_pairwise_accuracy(model, pairs)
    metrics.update(user_metadata(config))
    with (artifact_dir / "pairwise_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
