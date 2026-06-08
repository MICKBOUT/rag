VENV		= .venv
SRC_DIR		= src
MAIN		= $(SRC_DIR)/main.py

install:
	uv sync

run:
	uv run $(MAIN)

searching_one_dataset:
	uv run python -m student search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results

evaluate_search_results:
	uv run python -m moulinette evaluate_student_search_results --student_answer_path data/output/search_results/dataset_docs_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json --k 10 --max_context_length 2000

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
	@rm -rf $(VENV) bm25s_index_llm data/output/* data/processed/*
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@echo "Project clean"
