FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY pyproject.toml README.md requirements.txt ./
COPY configs ./configs
COPY scripts ./scripts
COPY src ./src

CMD ["bash", "scripts/run_toy_pipeline.sh"]
