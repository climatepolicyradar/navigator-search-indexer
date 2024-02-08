include .env

.PHONY: build test dev_install

setup:
	cp .env.example .env

build:
	docker build -t navigator-search-indexer .

vespa_setup: vespa_confirm_cli_installed vespa_dev_start vespa_healthy vespa_deploy_schema

test:
	docker-compose -f docker-compose.dev.yml run --rm navigator-search-indexer python -m pytest -vvv

dev_install:
	poetry install && poetry run pre-commit install

# setup dev/test vespa
vespa_confirm_cli_installed:
	@if [ ! $$(which vespa) ]; then \
		echo 'ERROR: The vespa cli is not installed, please install and try again:' ; \
		echo 'https://docs.vespa.ai/en/vespa-cli.html'; \
		exit 1; \
	fi

vespa_dev_start:
	docker compose -f docker-compose.dev.yml up --detach --wait vespaindexertest

vespa_healthy:
	@if [ ! $$(curl -f -s 'http://localhost:19071/status.html') ]; then \
		echo 'ERROR: Bad response from local vespa cluster, is it running?'; \
		exit 1; \
	fi

.ONESHELL:
vespa_deploy_schema:
	vespa config set target local
	@vespa deploy tests/vespa_test_schema --wait 300
