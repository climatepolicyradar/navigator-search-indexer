.PHONY: build test dev_install opensearch_test_data

setup:
	cp .env.example .env

build:
	docker build -t navigator-search-indexer .

run_encoding_docker:
	docker run --entrypoint python -v ${PWD}/data:/app/data navigator-search-indexer -m cli.text2embeddings ./data/raw ./data/processed

run_indexing_docker:
	docker run --entrypoint python --env-file=.env -v ${PWD}/data:/app/data aws s3 sync ${INDEXER_INPUT_PREFIX} /app/data/processed
	docker run --entrypoint python --network=host --env-file=.env -v ${PWD}/data:/app/data navigator-search-indexer -m cli.index_data /app/data/processed

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
	docker run --entrypoint python --network=host --env-file=.env -e OPENSEARCH_INDEX_PREFIX=navigator_test -v ${PWD}/data:/app/data navigator-search-indexer -m cli.test.create_test_index /app/data/cpr-dev-data-pipeline-cache/embeddings_input

opensearch_test_dump: create_test_index
	rm -rf ./data/opensearch_test_dump/**
	multielasticdump --input=http://admin:admin@localhost:9200 --output=./data/opensearch_test_dump --match="navigator_test_.*" --ignoreType=template
