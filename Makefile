# Convenience wrappers around the commands in RUNBOOK.md.
# Every target here maps to exactly one runbook block, so the two never drift.

PYTHON ?= python

.PHONY: help install check ingest ingest-all versions api ui watcher test eval eval-judge \
        experiment-chunking experiment-topk reset

help:
	@grep -E '^[a-zA-Z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-22s %s\n", $$1, $$2}'

install:               ## Create the virtualenv contents (run inside an activated venv)
	$(PYTHON) -m pip install -r requirements.txt

check:                 ## Pre-flight: config, Ollama, models, LangSmith, index
	$(PYTHON) scripts/check_environment.py

ingest:                ## Ingest all enabled sources with the active chunking strategy
	$(PYTHON) scripts/ingest.py

ingest-all:            ## Ingest into every chunking strategy (needed before the chunking sweep)
	$(PYTHON) scripts/ingest.py --all-strategies

versions:              ## Show the version history of the local policy
	$(PYTHON) scripts/simulate_policy_update.py --status

api:                   ## Run the FastAPI backend on :8000
	$(PYTHON) run_backend.py --reload

ui:                    ## Run the Streamlit frontend on :8501
	streamlit run frontend/streamlit_app.py

watcher:               ## Run the long-lived document update process
	$(PYTHON) scripts/run_watcher.py

test:                  ## Run the test suite (fully offline, no Ollama needed)
	$(PYTHON) -m pytest

eval:                  ## Run the golden dataset
	$(PYTHON) evaluation/run_evaluation.py

eval-judge:            ## Run the golden dataset with model-scored faithfulness
	$(PYTHON) evaluation/run_evaluation.py --judge

experiment-chunking:   ## Compare every chunking strategy
	$(PYTHON) evaluation/experiments.py chunking

experiment-topk:       ## Compare retrieval depth
	$(PYTHON) evaluation/experiments.py top-k --values 3 5 8 10

reset:                 ## Delete the local index and version history
	$(PYTHON) scripts/reset_data.py --yes
