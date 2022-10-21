.PHONY: build test dev_install
include .env

setup:
	cp .env.example .env

build:
	cp Dockerfile.local.example Dockerfile
	docker build -t navigator-search-indexer .

run_encoding_docker:
	docker run -v ${PWD}/data:/app/data navigator-search-indexer python -m cli.text2embeddings ./data/raw ./data/processed

run_indexing_docker:
	docker run --network=host --env-file=.env -v ${PWD}/data:/app/data navigator-search-indexer python -m cli.index_data ./data/processed

test:
	docker run navigator-search-indexer python -m pytest

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

build_and_push_ecr:
	aws ecr get-login-password --region eu-west-2 --profile dev | docker login --username AWS --password-stdin ${ECR_REPO}
	cp Dockerfile.aws.example Dockerfile
	docker build -t navigator-search-indexer-staging .
	docker tag navigator-search-indexer-staging:latest ${ECR_REPO}/navigator-search-indexer-staging:latest
	docker push ${ECR_REPO}/navigator-search-indexer-staging:latest