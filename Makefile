.PHONY: build test

build:
	docker build -t navigator-search-indexer .

test:
	docker run navigator-search-indexer python -m pytest