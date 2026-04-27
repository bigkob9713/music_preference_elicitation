#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-music-pref-elicitation:muqeval}"

cd "${ROOT_DIR}"

echo "Building MuQ-Eval Docker image: ${IMAGE_NAME}"
docker build -f Dockerfile.muqeval -t "${IMAGE_NAME}" .

