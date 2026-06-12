VENV		= .venv
SRC_DIR		= src
MAIN		= $(SRC_DIR)/main.py
PRIVACY=public

install:
	uv sync
	unzip data/raw/vllm-0.10.1.zip -d data/raw

run:
	uv run python -m student $(ARGS)

index:
	uv run python -m student index

evaluate_search_results:
	uv run python -m student search_dataset data/datasets/$(PRIVACY)/UnansweredQuestions/dataset_docs_$(PRIVACY).json --k 10 --save_directory data/output/search_results
	./moulinette/moulinette_pkg/moulinette-ubuntu evaluate_student_search_results --student_answer_path data/output/search_results/dataset_docs_$(PRIVACY).json --dataset_path data/datasets/$(PRIVACY)/AnsweredQuestions/dataset_docs_$(PRIVACY).json --k 10 --max_context_length 2000

evaluate_code_results:
	uv run python -m student search_dataset data/datasets/$(PRIVACY)/UnansweredQuestions/dataset_code_$(PRIVACY).json --k 10 --save_directory data/output/search_results --max_chunk_size 2000
	./moulinette/moulinette_pkg/moulinette-ubuntu evaluate_student_search_results --student_answer_path data/output/search_results/dataset_code_$(PRIVACY).json data/datasets/$(PRIVACY)/AnsweredQuestions/dataset_code_$(PRIVACY).json --k 10 --max_context_length 2000

recall_code:
	uv run python -m student search_dataset data/datasets/$(PRIVACY)/UnansweredQuestions/dataset_code_$(PRIVACY).json --k 10 --save_directory data/output/search_results --max_chunk_size 2000
	uv run python3 -m student evaluate data/output/search_results/dataset_code_$(PRIVACY).json data/datasets/$(PRIVACY)/AnsweredQuestions/dataset_code_$(PRIVACY).json --threshold 0.5

recall_docs:
	uv run python -m student search_dataset data/datasets/$(PRIVACY)/UnansweredQuestions/dataset_docs_$(PRIVACY).json --k 10 --save_directory data/output/search_results --max_chunk_size 2000
	uv run python3 -m student evaluate data/output/search_results/dataset_docs_$(PRIVACY).json data/datasets/$(PRIVACY)/AnsweredQuestions/dataset_docs_$(PRIVACY).json --threshold 0.8

answer_the_dataset:
	uv run python -m student search_dataset data/datasets/public/UnansweredQuestions/dataset_code_public.json \
		--save_directory data/output/search_results
	uv run python -m student answer_dataset \
		--student_search_results_path data/output/search_results/dataset_code_public.json \
		--save_directory data/output/search_results_and_answer \
		--model Qwen/Qwen3-0.6B

answer_model_2:
	uv run python -m student answer "What's the default value of trust_remote_code in vLLM's LLM class constructor?" --model Qwen/Qwen3-1.7B


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
	@rm -rf $(VENV) bm25s_index_llm data/output/* data/processed/* data/raw/vllm-0.10.1
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@echo "Project clean"
