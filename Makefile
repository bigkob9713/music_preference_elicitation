PYTHON ?= python3
CONFIG ?= configs/toy_major_minor.yaml

.PHONY: setup data train eval-pairwise eval-rerank run clean

setup:
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && python -m pip install --upgrade pip setuptools wheel
	. .venv/bin/activate && python -m pip install -r requirements.txt
	. .venv/bin/activate && python -m pip install -e .

data:
	PYTHONPATH=. $(PYTHON) -m src.data.make_synthetic_pairs --config $(CONFIG)

train:
	PYTHONPATH=. $(PYTHON) -m src.train.train_preference --config $(CONFIG)

eval-pairwise:
	PYTHONPATH=. $(PYTHON) -m src.eval.eval_pairwise --config $(CONFIG)

eval-rerank:
	PYTHONPATH=. $(PYTHON) -m src.eval.eval_rerank --config $(CONFIG)

run:
	bash scripts/run_toy_pipeline.sh $(CONFIG)

clean:
	rm -rf .venv build dist *.egg-info

