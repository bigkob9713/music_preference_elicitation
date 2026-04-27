#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-music-pref-elicitation:muqeval}"
CONFIG_PATH="${1:-configs/muqeval_bridge_small.yaml}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to your external MuQ-Eval bridge data root}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${DATA_ROOT}/hf_cache}"
DOCKER_GPU_ARGS="${DOCKER_GPU_ARGS:---gpus all}"

cd "${ROOT_DIR}"

echo "Running MuQ-Eval bridge image: ${IMAGE_NAME}"
echo "Config: ${CONFIG_PATH}"
echo "Data root: ${DATA_ROOT}"
echo "HF cache: ${HF_CACHE_DIR}"
echo "GPU args: ${DOCKER_GPU_ARGS}"

mkdir -p artifacts/muqeval_bridge_small

docker run --rm ${DOCKER_GPU_ARGS} \
  -v "${ROOT_DIR}:/workspace" \
  -v "${DATA_ROOT}:${DATA_ROOT}" \
  -e HF_HOME="${HF_CACHE_DIR}" \
  -e HF_HUB_CACHE="${HF_CACHE_DIR}/hub" \
  -w /workspace \
  "${IMAGE_NAME}" \
  python scripts/run_muqeval_bridge.py --config "${CONFIG_PATH}"

