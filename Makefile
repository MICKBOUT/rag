VENV		= .venv
SRC_DIR		= src
MAIN		= $(SRC_DIR)/main.py

install:
	uv sync

run:
	uv run $(MAIN)

lint: 
	uv run flake8 $(SRC_DIR)
	uv run mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 $(SRC_DIR)
	uv run mypy --strict

debug:
	uv run python -m pdb $(MAIN)

clean:
	@echo "cleaning project..."
	@uv clean
	@rm -rf $(VENV)
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@echo "Project clean"
