.PHONY: build test dev_install

build:
	docker build -t navigator-search-indexer .

test:
	docker run navigator-search-indexer python -m pytest

dev_install:
	poetry install && poetry run pre-commit install