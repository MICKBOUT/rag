VENV		= .venv
SRC_DIR		= src
MAIN		= $(SRC_DIR)/main.py

install:
	uv sync

run:
	uv run python -m student $(ARGS)

evaluate_search_results:
	uv run python -m student search_dataset --dataset_path data/datasets/private/UnansweredQuestions/dataset_docs_private.json --k 10 --save_directory data/output/search_results
	./moulinette/moulinette_pkg/moulinette-ubuntu evaluate_student_search_results --student_answer_path data/output/search_results/dataset_docs_private.json --dataset_path data/datasets/private/AnsweredQuestions/dataset_docs_private.json --k 10 --max_context_length 2000

evaluate_code_results:
	uv run python -m student search_dataset --dataset_path data/datasets/private/UnansweredQuestions/dataset_code_private.json --k 10 --save_directory data/output/search_results --max_chunk_size 2000
	./moulinette/moulinette_pkg/moulinette-ubuntu evaluate_student_search_results --student_answer_path data/output/search_results/dataset_code_private.json --dataset_path data/datasets/private/AnsweredQuestions/dataset_code_private.json --k 10 --max_context_length 2000

recall_code:
	uv run python -m student search_dataset --dataset_path data/datasets/private/UnansweredQuestions/dataset_code_private.json --k 10 --save_directory data/output/search_results --max_chunk_size 2000
	uv run python3 -m student evaluate data/output/search_results/dataset_code_private.json data/datasets/private/AnsweredQuestions/dataset_code_private.json

recall_docs:
	uv run python -m student search_dataset --dataset_path data/datasets/private/UnansweredQuestions/dataset_docs_private.json --k 10 --save_directory data/output/search_results --max_chunk_size 2000
	uv run python3 -m student evaluate data/output/search_results/dataset_docs_private.json data/datasets/private/AnsweredQuestions/dataset_docs_private.json

answer_the_dataset:
	uv run python -m student answer_dataset \
		--student_search_results_path data/output/search_results/dataset_docs_private.json \
		--save_directory data/output/search_results_and_answer \
		# --max_tokens 128

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
