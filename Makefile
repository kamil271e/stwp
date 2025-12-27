.PHONY: install install-dev format lint typecheck test test-cov api docker-build docker-up docker-down docker-logs clean help

.DEFAULT_GOAL := help

# ============================================================================
# Installation
# ============================================================================

install:
	uv sync

install-dev:
	uv sync --all-extras
	uv run pre-commit install

# ============================================================================
# Code Quality
# ============================================================================

format:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff check .

typecheck:
	uv run mypy src/

check: lint typecheck

# ============================================================================
# Testing
# ============================================================================

test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ -v --cov=src/stwp --cov-report=term-missing --cov-report=html

test-unit:
	uv run pytest tests/unit/ -v

test-api:
	uv run pytest tests/api/ -v

# ============================================================================
# Application
# ============================================================================

api:
	uv run python -m stwp.api.main

# ============================================================================
# Docker
# ============================================================================

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

docker-run: docker-build docker-up

# ============================================================================
# Cleanup
# ============================================================================

clean:
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf dist
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ============================================================================
# Help
# ============================================================================

help:
	@echo "STWP - Short Term Weather Prediction"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
