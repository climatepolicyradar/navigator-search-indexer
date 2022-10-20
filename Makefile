.PHONY: build test dev_install

build:
	cp Dockerfile.local.example Dockerfile
	docker build -t navigator-search-indexer .
	docker tag navigator-search-indexer:latest navigator-search-indexer-staging:latest

run_encoding_docker:
	docker run -v ${PWD}/data:/app/data navigator-search-indexer python -m cli.text2embeddings ./data/raw ./data/processed

run_indexing_docker:
	docker run --network=host --env-file=.env -v ${PWD}/data:/app/data navigator-search-indexer python -m cli.index_data ./data/processed

test:
	docker run navigator-search-indexer python -m pytest -vvv

dev_install:
	poetry install && poetry run pre-commit install

test_against_aws:
	cp Dockerfile.aws.example Dockerfile
	docker build -t navigator-search-indexer-aws .
	docker run -it navigator-search-indexer-aws python -m pytest

run_local_against_aws:
	cp Dockerfile.aws.example Dockerfile
	docker build -t navigator-search-indexer-aws .
	docker run -e parser_output_s3=s3://data-pipeline-a64047a/unit_tests/runs/parser_output/ -e indexer_output_s3=s3://data-pipeline-a64047a/unit_tests/runs/indexer_output/ -it navigator-search-indexer-aws

build_and_push_ecr:
	aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 281621126254.dkr.ecr.us-east-1.amazonaws.com
	cp Dockerfile.aws.example Dockerfile
	docker build -t indexer-4d7db1c .
	docker tag indexer-4d7db1c:latest 281621126254.dkr.ecr.us-east-1.amazonaws.com/indexer-4d7db1c:latest
	docker push 281621126254.dkr.ecr.us-east-1.amazonaws.com/indexer-4d7db1c:latest
