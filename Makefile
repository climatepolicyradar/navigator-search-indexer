include .env

build:
	docker build -t navigator-embeddings-generation .

run_embeddings_generation:
	docker run --entrypoint python -v ${PWD}/data:/app/data navigator-embeddings-generation -m cli.text2embeddings ./data/embeddings_input ./data/indexer_input

test:
	docker run --entrypoint python navigator-embeddings-generation -m pytest -vvv

dev_install:
	poetry install && poetry run pre-commit install
