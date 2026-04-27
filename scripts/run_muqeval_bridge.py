#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path


REQUIRED_MANIFEST_FIELDS = ("set_id", "candidate_id", "audio_path")


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_dir(path):
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_manifest(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = [field for field in REQUIRED_MANIFEST_FIELDS if field not in row]
            if missing:
                raise ValueError(f"Manifest row {line_number} missing fields: {missing}")
            rows.append(row)
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def load_muqeval_scorer(config):
    scorer_config = config["scorer"]
    repo_dir = scorer_config.get("muqeval_repo_dir", os.environ.get("MUQEVAL_REPO_DIR", "/opt/MuQ-Eval"))
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    import torch
    import numpy as np
    import soundfile as sf
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf
    from src.data import AudioProcessor
    from src.model import MusicQualityModel

    requested_device = scorer_config.get("device", "auto")
    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = requested_device

    cache_dir = scorer_config.get("hf_cache_dir")
    base_config_path = hf_hub_download(
        scorer_config["hf_repo_id"],
        scorer_config.get("base_config_filename", "base.yaml"),
        cache_dir=cache_dir,
    )
    config_path = hf_hub_download(
        scorer_config["hf_repo_id"],
        scorer_config.get("config_filename", "config.yaml"),
        cache_dir=cache_dir,
    )
    model_path = hf_hub_download(
        scorer_config["hf_repo_id"],
        scorer_config.get("model_filename", "model_state_dict.pt"),
        cache_dir=cache_dir,
    )

    base_cfg = OmegaConf.load(base_config_path)
    override_cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(base_cfg, override_cfg)
    model = MusicQualityModel(cfg)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = extract_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    sample_rate = int(scorer_config.get("sample_rate", 24000))
    max_duration = float(scorer_config.get("max_duration", 10.0))
    processor = AudioProcessor(
        target_sr=sample_rate,
        clip_samples=int(sample_rate * max_duration),
    )

    def score_audio(audio_path):
        array, source_sr = sf.read(audio_path)
        if array.ndim > 1:
            array = array.mean(axis=1)
        waveform = processor.process(
            np.array(array, dtype=np.float32),
            source_sr,
            mode="center",
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            scores = model(waveform)
        return {key: float(value.item()) for key, value in scores.items()}

    return score_audio, device


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint

    for key in ("model_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return strip_module_prefix(value)

    return strip_module_prefix(checkpoint)


def strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(isinstance(key, str) and key.startswith("module.") for key in state_dict):
        return state_dict
    return {
        key[len("module."):] if isinstance(key, str) and key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def score_candidates(config, manifest_rows):
    score_audio, device = load_muqeval_scorer(config)
    score_key = config.get("score_key", "MI")
    scored_rows = []

    for row in manifest_rows:
        audio_path = row["audio_path"]
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        raw_scores = score_audio(audio_path)
        if score_key not in raw_scores:
            raise KeyError(f"Score key {score_key!r} not found in MuQ-Eval output keys: {sorted(raw_scores)}")
        scored = dict(row)
        scored["score"] = raw_scores[score_key]
        scored["score_key"] = score_key
        scored["raw_scores"] = raw_scores
        scored["device"] = device
        scored_rows.append(scored)

    return scored_rows


def group_by_set(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["set_id"]].append(row)
    return grouped


def evaluate_reranking(config, scored_rows):
    rng = random.Random(config["seed"])
    grouped = group_by_set(scored_rows)
    ranked_rows = []
    input_scores = []
    random_scores = []
    score_ranked_scores = []

    for set_id, candidates in grouped.items():
        input_top = candidates[0]
        random_top = rng.choice(candidates)
        ranked = sorted(candidates, key=lambda row: row["score"], reverse=True)
        score_top = ranked[0]

        input_scores.append(input_top["score"])
        random_scores.append(random_top["score"])
        score_ranked_scores.append(score_top["score"])

        for rank, row in enumerate(ranked, start=1):
            ranked_row = dict(row)
            ranked_row["rank"] = rank
            ranked_rows.append(ranked_row)

    total_sets = max(1, len(grouped))
    total_candidates = len(scored_rows)
    metrics = {
        "num_sets": len(grouped),
        "num_candidates": total_candidates,
        "score_extraction_success_rate": total_candidates / max(1, total_candidates),
        "input_order_top1_score": sum(input_scores) / total_sets,
        "random_top1_score": sum(random_scores) / total_sets,
        "score_ranked_top1_score": sum(score_ranked_scores) / total_sets,
        "score_ranked_improvement_over_input": (sum(score_ranked_scores) - sum(input_scores)) / total_sets,
        "score_ranked_improvement_over_random": (sum(score_ranked_scores) - sum(random_scores)) / total_sets,
    }
    return metrics, ranked_rows


def validate_pilot_shape(config, manifest_rows):
    expected_num_sets = config.get("expected_num_sets")
    expected_candidates_per_set = config.get("expected_candidates_per_set")
    grouped = group_by_set(manifest_rows)
    if expected_num_sets is not None and len(grouped) != int(expected_num_sets):
        raise ValueError(f"Expected {expected_num_sets} set(s), found {len(grouped)}")
    if expected_candidates_per_set is not None:
        for set_id, candidates in grouped.items():
            if len(candidates) != int(expected_candidates_per_set):
                raise ValueError(
                    f"Expected {expected_candidates_per_set} candidate(s) in set {set_id}, "
                    f"found {len(candidates)}"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    manifest_rows = read_manifest(config["manifest_path"])
    validate_pilot_shape(config, manifest_rows)

    artifact_dir = ensure_dir(config["artifact_dir"])
    scored_rows = score_candidates(config, manifest_rows)
    metrics, ranked_rows = evaluate_reranking(config, scored_rows)

    write_jsonl(artifact_dir / "scores.jsonl", scored_rows)
    write_jsonl(artifact_dir / "ranked_candidates.jsonl", ranked_rows)
    with open(artifact_dir / "rerank_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
