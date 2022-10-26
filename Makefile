.PHONY: build test dev_install
include .env

setup:
	cp .env.example .env

build:
	docker build -t navigator-search-indexer .

run_encoding_docker:
	docker run  --entrypoint python -v ${PWD}/data:/app/data navigator-search-indexer -m cli.text2embeddings ./data/raw ./data/processed

run_indexing_docker:
	docker run --entrypoint python --network=host --env-file=.env -v ${PWD}/data:/app/data navigator-search-indexer -m cli.index_data ./data/processed

test:
	docker run --entrypoint python navigator-search-indexer -m pytest -vvv

dev_install:
	poetry install && poetry run pre-commit install

test_against_aws:
	cp Dockerfile.aws.example Dockerfile
	docker build -t navigator-search-indexer-aws .
	docker run -it navigator-search-indexer-aws python -m pytest

run_local_against_aws:
	cp Dockerfile.aws.example Dockerfile
	docker build -t navigator-search-indexer-aws .
	docker run -e EMBEDDINGS_INPUT_PREFIX=${EMBEDDINGS_INPUT_PREFIX} -e INDEXER_INPUT_PREFIX=${INDEXER_INPUT_PREFIX} -it navigator-search-indexer-aws
