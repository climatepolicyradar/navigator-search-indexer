.PHONY: build test dev_install

build:
	docker build -t navigator-search-indexer .

run_encoding_docker:
	docker run -v ${PWD}/data:/app/data navigator-search-indexer python -m cli.text2embeddings ./data/raw ./data/processed

run_indexing_docker:
	docker run --network=host --env-file=.env -v ${PWD}/data:/app/data navigator-search-indexer python -m cli.index_data ./data/processed

test:
	docker run navigator-search-indexer python -m pytest

dev_install:
	poetry install && poetry run pre-commit install