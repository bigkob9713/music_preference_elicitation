#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-music-pref-elicitation:toy}"
CONFIG_PATH="${1:-configs/toy_major_minor.yaml}"
DOCKER_GPU_ARGS="${DOCKER_GPU_ARGS:---gpus all}"

cd "${ROOT_DIR}"

echo "Running Docker image: ${IMAGE_NAME}"
echo "Config: ${CONFIG_PATH}"
echo "GPU args: ${DOCKER_GPU_ARGS}"

docker run --rm ${DOCKER_GPU_ARGS} \
  -v "${ROOT_DIR}:/workspace" \
  -w /workspace \
  "${IMAGE_NAME}" \
  bash scripts/run_toy_pipeline.sh "${CONFIG_PATH}"

