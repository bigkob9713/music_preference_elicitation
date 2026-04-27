#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-${ROOT_DIR}/configs/toy_major_minor.yaml}"

export PYTHONPATH="${ROOT_DIR}"

python3 -m src.data.make_synthetic_pairs --config "${CONFIG_PATH}"
python3 -m src.train.train_preference --config "${CONFIG_PATH}"
python3 -m src.eval.eval_pairwise --config "${CONFIG_PATH}"
python3 -m src.eval.eval_rerank --config "${CONFIG_PATH}"

