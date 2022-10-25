#!/bin/bash
set -e

python -m cli.text2embeddings --s3 --device=cpu "${EMBEDDINGS_INPUT_PREFIX}" "${INDEXER_INPUT_PREFIX}" "${FILES_TO_PARSE}"