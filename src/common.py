import json
from pathlib import Path
from typing import Dict, List


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_dir(path: str) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def user_metadata(config: Dict) -> Dict:
    user_config = config.get("user", {})
    metadata = {"user_type": user_config.get("type", "deterministic")}
    if "major_utility" in user_config:
        metadata["major_utility"] = user_config["major_utility"]
    if "minor_utility" in user_config:
        metadata["minor_utility"] = user_config["minor_utility"]
    if "tempo_weight" in user_config:
        metadata["tempo_weight"] = user_config["tempo_weight"]
    return metadata
