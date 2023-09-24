#!/bin/bash
set -e

python -m cli.index_data --s3 "${INDEXER_INPUT_PREFIX}"