import argparse
import json
from pathlib import Path

from src.common import load_config, read_jsonl
from src.models.preference_model import LinearPreferenceModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    artifact_dir = Path(config["artifact_dir"])
    train_pairs = read_jsonl(artifact_dir / "train_pairs.jsonl")

    model = LinearPreferenceModel()
    history = model.train(
        train_pairs,
        learning_rate=float(config["model"]["learning_rate"]),
        epochs=int(config["model"]["epochs"]),
    )

    model.save(artifact_dir / "preference_model.json")
    with (artifact_dir / "train_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    final = history[-1]
    print(f"Trained on {len(train_pairs)} pairs")
    print(f"Final avg loss: {final['avg_loss']:.4f}")
    print(f"Weights: {final['weights']}")
    print(f"Bias: {final['bias']:.4f}")


if __name__ == "__main__":
    main()

