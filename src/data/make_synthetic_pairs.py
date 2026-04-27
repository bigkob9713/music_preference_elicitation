import argparse
import random
from pathlib import Path
from typing import Dict, List

from src.common import ensure_dir, load_config, write_jsonl
from src.users.pseudo_user_major import build_major_pseudo_user


def build_candidate(candidate_id: str, rng: random.Random) -> Dict:
    is_major = rng.randint(0, 1)
    return {
        "id": candidate_id,
        "mode": "major" if is_major else "minor",
        "features": {
            "is_major": float(is_major),
            "tempo_norm": round(rng.random(), 6),
            "energy": round(rng.random(), 6),
        },
    }


def make_preference_pairs(num_pairs: int, rng: random.Random, user) -> List[Dict]:
    pairs = []
    while len(pairs) < num_pairs:
        left = build_candidate(f"cand_{len(pairs)}_a", rng)
        right = build_candidate(f"cand_{len(pairs)}_b", rng)
        preference = user.prefer(left, right)
        if preference == 0:
            continue
        preferred = left if preference > 0 else right
        other = right if preference > 0 else left
        pairs.append({"preferred": preferred, "other": other})
    return pairs


def make_rerank_sets(
    num_sets: int,
    candidates_per_set: int,
    rng: random.Random,
) -> List[Dict]:
    rerank_sets = []
    for set_idx in range(num_sets):
        candidates = [
            build_candidate(f"set_{set_idx}_cand_{cand_idx}", rng)
            for cand_idx in range(candidates_per_set)
        ]
        if not any(candidate["mode"] == "major" for candidate in candidates):
            candidates[0]["mode"] = "major"
            candidates[0]["features"]["is_major"] = 1.0
        if not any(candidate["mode"] == "minor" for candidate in candidates):
            candidates[0]["mode"] = "minor"
            candidates[0]["features"]["is_major"] = 0.0
        rng.shuffle(candidates)
        rerank_sets.append({"set_id": f"set_{set_idx}", "candidates": candidates})
    return rerank_sets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    artifact_dir = ensure_dir(config["artifact_dir"])
    rng = random.Random(config["seed"])
    user = build_major_pseudo_user(config, rng)

    train_pairs = make_preference_pairs(config["data"]["train_pairs"], rng, user)
    test_pairs = make_preference_pairs(config["data"]["test_pairs"], rng, user)
    rerank_sets = make_rerank_sets(
        config["data"]["rerank_sets"],
        config["data"]["candidates_per_set"],
        rng,
    )

    write_jsonl(Path(artifact_dir) / "train_pairs.jsonl", train_pairs)
    write_jsonl(Path(artifact_dir) / "test_pairs.jsonl", test_pairs)
    write_jsonl(Path(artifact_dir) / "rerank_sets.jsonl", rerank_sets)

    print(f"Wrote train pairs: {len(train_pairs)}")
    print(f"Wrote test pairs: {len(test_pairs)}")
    print(f"Wrote rerank sets: {len(rerank_sets)}")
    print(f"User type: {config.get('user', {}).get('type', 'deterministic')}")
    print(f"Artifacts: {artifact_dir}")


if __name__ == "__main__":
    main()
