include .env

.PHONY: build test dev_install opensearch_test_data

setup:
	cp .env.example .env

build:
	docker build -t navigator-search-indexer .

download_indexer_inputs:
	docker run --entrypoint aws --env-file=.env -v ${PWD}/data:/app/data navigator-search-indexer s3 sync ${INDEXER_INPUT_PREFIX} /app/data/indexer_input

run_encoding_docker:
	docker run --entrypoint python -v ${PWD}/data:/app/data navigator-search-indexer -m cli.text2embeddings ./data/embeddings_input ./data/indexer_input

run_indexing_docker: download_indexer_inputs
	docker run --entrypoint python --network=host --env-file=.env -v ${PWD}/data:/app/data navigator-search-indexer -m cli.index_data /app/data/indexer_input

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

# test data for backend
create_test_index:
	docker run --entrypoint python --network=host --env-file=.env -e OPENSEARCH_INDEX_PREFIX=navigator_test -v ${PWD}/data:/app/data navigator-search-indexer -m cli.test.create_test_index /app/data/embeddings_input

opensearch_test_dump: create_test_index
	rm -rf ./data/opensearch_test_dump/**
	multielasticdump --input=http://admin:admin@localhost:9200 --output=./data/opensearch_test_dump --match="navigator_test_.*" --ignoreType=template
