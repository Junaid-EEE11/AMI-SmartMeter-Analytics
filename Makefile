# Makefile for shift-ami: Trustworthy AMI Forecasting Under Behavioral Distribution Shift

.PHONY: help install test prepare smoke experiment ablations report clean

help:
	@echo "Available commands:"
	@echo "  make install     - Install development dependencies in editable mode"
	@echo "  make test        - Run complete pytest suite with strict leakage checks"
	@echo "  make prepare     - Prepare cohort aggregates and data quality report"
	@echo "  make smoke       - Run fast end-to-end smoke test pipeline"
	@echo "  make experiment  - Run master experiment with paired block bootstrap"
	@echo "  make ablations   - Run comprehensive ablation experiments (A1-A8)"
	@echo "  make report      - Build all tables, figures, failure analysis, and reports"
	@echo "  make clean       - Remove cached artifacts and temporary build files"

install:
	python -m pip install -e .[dev]

test:
	pytest -v

prepare:
	python scripts/prepare_data.py --config configs/main.yaml

smoke:
	python scripts/run_experiment.py --config configs/smoke.yaml

experiment:
	python scripts/run_experiment.py --config configs/main.yaml

ablations:
	python scripts/run_ablations.py --config configs/main.yaml

report:
	python scripts/build_report.py --config configs/main.yaml

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov build dist *.egg-info
