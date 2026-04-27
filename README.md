# Music Preference Learning and Reranking

This repository is a research scaffold for **music preference learning and candidate reranking**.

The long-term motivation is to study how a system can recover a user's evaluation criterion and use it to rank music candidates more effectively. Rather than jumping directly to a full interactive system, this repository takes a staged approach:

1. **deterministic toy preference learning**
2. **stochastic utility-based preference learning**
3. **multi-factor reranking with utility-aware evaluation**
4. **a minimal bridge to real-audio scoring with MuQ-Eval**

The focus is not realism yet. The focus is to build a clean empirical path from simple synthetic settings to small real-audio scorer integration.

## Why this repository exists

Preference learning for creative domains like music is difficult because:

- user preferences are often uncertain or evolving
- simple top-1 accuracy can hide important failures
- realistic audio pipelines are expensive and hard to debug

This repository addresses that by starting with small, controlled experiments and gradually increasing realism.

The main design principle is:

> **Build the smallest setting that makes the next research question testable.**

## What is implemented

### 1. Deterministic toy baseline
A simple pseudo-user always prefers `major` over `minor`.

This stage is used to verify that:
- pairwise preference learning works
- reranking works end-to-end
- oracle-like behavior is reachable in a clean setting

### 2. Stochastic single-factor pseudo-user
A pseudo-user is defined through a **utility-based Bradley-Terry / logistic choice model** instead of deterministic labels.

This stage is used to verify that:
- the model can handle probabilistic preferences
- probability-aware metrics such as log loss or Brier score become meaningful
- reranking can still recover high-utility candidates

### 3. Stochastic two-factor pseudo-user
The pseudo-user utility includes:
- a primary factor: `major/minor`
- a weak secondary factor: `tempo_norm`

This stage is important because it shows that:
- **major-rate metrics can saturate**
- while **expected utility and regret still reveal meaningful reranking errors**

In other words, this repository already includes a toy setting where simple major/minor success metrics are no longer enough, and utility-aware evaluation becomes necessary.

### 4. Minimal MuQ-Eval bridge
The repository also includes a minimal bridge experiment that treats **MuQ-Eval** as a fixed external scorer on a small set of real audio candidates.

The purpose of this bridge is not full preference modeling. It is to verify that the repository can:
- score real audio candidates with an external model
- save per-candidate scores
- compare input-order, random, and score-based reranking

This gives a small but working bridge from synthetic toy worlds to real-audio scorer integration.

## Repository structure

- `configs/` — experiment configs
- `src/data/` — synthetic pair and candidate-set generation
- `src/users/` — pseudo-user definitions
- `src/models/` — small preference models
- `src/train/` — training entrypoints
- `src/eval/` — evaluation scripts
- `scripts/` — end-to-end run scripts and Docker helpers

## Toy setting

The synthetic candidates use simple feature vectors such as:

- `is_major`
- `tempo_norm`
- `energy`

The toy experiments intentionally start with very small, interpretable representations so that failures are easy to diagnose.

## Evaluation philosophy

For deterministic settings, basic accuracy-style metrics are often enough.

For stochastic and multi-factor settings, this repository emphasizes **utility-aware reranking metrics**, including:

- held-out pairwise accuracy
- log loss or Brier score
- top-1 expected utility
- regret relative to oracle

Major-rate metrics are still useful as sanity checks, but they are treated as **secondary** once the preference world becomes stochastic or multi-factor.

## Quick start

### Run the default toy pipeline

```bash
bash scripts/run_toy_pipeline.sh
```

This runs:

1. synthetic data generation
2. preference-model training
3. pairwise evaluation
4. reranking evaluation

Artifacts are written under `artifacts/`.

You can also run a specific config:

```bash
bash scripts/run_toy_pipeline.sh <config-path>
```

### Example toy configs

Examples under `configs/` include:

* deterministic toy baseline
* stochastic single-factor baseline
* stochastic two-factor baseline

## Docker workflow

This repository is designed to support a simple local-development / remote-execution workflow:

* edit code locally
* push to GitHub
* pull on a remote Linux server
* run experiments there through Docker

Build the standard toy image:

```bash
bash scripts/docker_build.sh
```

Run the toy pipeline:

```bash
bash scripts/docker_run_toy.sh
```

## MuQ-Eval bridge

A separate Docker path is provided for the MuQ-Eval bridge experiment so that the synthetic toy baseline stays dependency-light.

Build the MuQ-Eval bridge image:

```bash
bash scripts/docker_build_muqeval.sh
```

Run the MuQ-Eval bridge:

```bash
bash scripts/docker_run_muqeval_bridge.sh
```

The current bridge is intentionally minimal:

* fixed audio candidate sets
* fixed external scorer
* input-order vs random vs score-based reranking

### MuQ-Eval bridge prerequisites

The MuQ-Eval bridge requires a small external audio candidate set and a JSONL manifest.
These audio files are **not included in this repository**.

At minimum, each manifest row should contain:

- `set_id`
- `candidate_id`
- `audio_path`

Optional fields such as `prompt` and `source` may also be included.

The current bridge is intended as a small pilot for fixed audio-candidate scoring and reranking, not as a fully packaged benchmark dataset.

### MuQ-Eval bridge config note

`configs/muqeval_bridge_small.yaml` contains placeholder absolute paths and is meant as an example config.

Before running the MuQ-Eval bridge, update it for your local or server environment so that the manifest path, cache path, and audio file locations are valid.

## Current scope

This repository currently emphasizes:

* staged empirical design
* preference learning in synthetic settings
* utility-aware reranking evaluation
* minimal real-audio scorer integration

It does **not** yet include:

* full interactive preference elicitation
<!-- * active query policies -->
* real human preference collection
* end-to-end text-to-music training

Those are future directions, not current claims.

## Why this project is interesting

I think the most interesting part of this repository is not any single model, but the **research workflow it encodes**:

* start with a clean deterministic baseline
* move to stochastic choice
* move to multi-factor utility
* notice when simple success metrics stop being informative
* switch to utility-aware evaluation
* build a bridge toward real-audio scoring without breaking the toy pipeline

That progression is the main contribution of the repository at its current stage.

## Status

At the current stage, the repository provides:

* a working deterministic toy pipeline
* stochastic single-factor and multi-factor toy baselines
* utility-aware reranking evaluation
* random and oracle baselines in toy settings
* a minimal working MuQ-Eval bridge on a small pilot setup

## Notes

This is an active research scaffold, not a finished benchmark or production system.

The current emphasis is:

* correctness
* interpretability
* small runnable experiments
* clean separation between toy baselines and real-audio bridge experiments
