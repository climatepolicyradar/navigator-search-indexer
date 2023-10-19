#!/bin/bash
set -e

if [[ -z ${VESPA_KEY_LOCATION} ]]; then
    echo "VESPA_KEY_LOCATION not found"
else
    mkdir -p $(dirname $VESPA_KEY_LOCATION)
    echo ${VESPA_PRIVATE_KEY} > ${VESPA_KEY_LOCATION}
fi

python -m cli.index_data --s3 "${INDEXER_INPUT_PREFIX}"
