#!/bin/bash
set -e

echo Setting up for vespa
args=$@
echo Running with args: $args

mkdir -p $(dirname $VESPA_KEY_LOCATION)
mkdir -p $(dirname $VESPA_CERT_LOCATION)

# These values where initially encoded from the key and cert with:
# $ openssl base64 -a -A -in <file>
echo "${VESPA_PRIVATE_KEY}" | openssl base64 -d -a -A --out ${VESPA_KEY_LOCATION}
echo "${VESPA_PUBLIC_CERT}" | openssl base64 -d -a -A --out ${VESPA_CERT_LOCATION}

python -m cli.index_data $args
