OPENSEARCH_USER=admin \
OPENSEARCH_PASSWORD=admin \
OPENSEARCH_URL=http://localhost:9200 \
OPENSEARCH_INDEX=navigator \
OPENSEARCH_USE_SSL=False \
OPENSEARCH_VERIFY_CERTS=False \
OPENSEARCH_SSL_WARNINGS=False \
python -m cli.index_data data/raw data/processed 