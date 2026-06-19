VENV		= .venv
SRC_DIR		= src
MAIN		= $(SRC_DIR)
PRIVACY=public

install:
	uv sync
	unzip data/raw/vllm-0.10.1.zip -d data/raw

run:
	uv run python -m $(SRC_DIR) $(ARGS)

index:
	uv run python -m $(SRC_DIR) index

evaluate_docs_results:
	uv run python -m $(SRC_DIR) search_dataset data/datasets/$(PRIVACY)/UnansweredQuestions/dataset_docs_$(PRIVACY).json
	./moulinette/moulinette-ubuntu evaluate_student_search_results --student_answer_path data/output/search_results/dataset_docs_$(PRIVACY).json --dataset_path data/datasets/$(PRIVACY)/AnsweredQuestions/dataset_docs_$(PRIVACY).json

evaluate_code_results:
	uv run python -m $(SRC_DIR) search_dataset data/datasets/$(PRIVACY)/UnansweredQuestions/dataset_code_$(PRIVACY).json
	./moulinette/moulinette-ubuntu evaluate_student_search_results --student_answer_path data/output/search_results/dataset_code_$(PRIVACY).json data/datasets/$(PRIVACY)/AnsweredQuestions/dataset_code_$(PRIVACY).json

recall_code:	
	uv run python -m $(SRC_DIR) search_dataset data/datasets/$(PRIVACY)/UnansweredQuestions/dataset_code_$(PRIVACY).json
	uv run python -m $(SRC_DIR) evaluate data/output/search_results/dataset_code_$(PRIVACY).json data/datasets/$(PRIVACY)/AnsweredQuestions/dataset_code_$(PRIVACY).json --threshold 0.5

recall_docs:
	uv run python -m $(SRC_DIR) search_dataset data/datasets/$(PRIVACY)/UnansweredQuestions/dataset_docs_$(PRIVACY).json
	uv run python -m $(SRC_DIR) evaluate data/output/search_results/dataset_docs_$(PRIVACY).json data/datasets/$(PRIVACY)/AnsweredQuestions/dataset_docs_$(PRIVACY).json --threshold 0.8

answer_the_dataset:
	uv run python -m $(SRC_DIR) search_dataset data/datasets/public/UnansweredQuestions/dataset_code_public.json \
		--save_directory data/output/search_results \
		--k 1

	uv run python -m $(SRC_DIR) answer_dataset \
		--student_search_results_path data/output/search_results/dataset_code_public.json \

answer_model_2:
	uv run python -m $(SRC_DIR) answer "What's the default value of trust_remote_code in vLLM's LLM class constructor?" --model Qwen/Qwen3-1.7B


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
