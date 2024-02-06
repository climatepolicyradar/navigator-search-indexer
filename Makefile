include .env

.PHONY: build test dev_install

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

# setup dev/test vespa
vespa_confirm_cli_installed:
	@if [ ! $$(which vespa) ]; then \
		echo 'ERROR: The vespa cli is not installed, please install and try again:' ; \
		echo 'https://docs.vespa.ai/en/vespa-cli.html'; \
		exit 1; \
	fi

vespa_dev_start:
	docker-compose -f docker-compose.dev.yml up -d --remove-orphans --wait

vespa_healthy:
	@if [ ! $$(curl -f -s 'http://localhost:19071/status.html') ]; then \
		echo 'ERROR: Bad response from local vespa cluster, is it running?'; \
		exit 1; \
	fi

.ONESHELL:
vespa_deploy_schema:
	vespa config set target local
	@vespa deploy tests/vespa_test_schema --wait 300

vespa_setup: vespa_confirm_cli_installed vespa_dev_start vespa_healthy vespa_deploy_schema
