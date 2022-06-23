# docker-compose run -v ~/Documents/CPR/data/_alpha-temp/final/ids_msmarco-distilbert-dot-v5_20220512-1415.json:/text-ids-path \
# -v ~/Documents/CPR/data/_alpha-temp/final/embeddings_dim_768_msmarco-distilbert-dot-v5_20220512-1415.memmap:/embeddings-path \
# -v ~/Documents/CPR/data/_alpha-temp/final/description_ids_msmarco-distilbert-dot-v5_20220512-1415.csv:/desc-ids-path \
# -v ~/Documents/CPR/data/_alpha-temp/final/description_embs_dim_768_msmarco-distilbert-dot-v5_20220512-1415.memmap:/desc-embeddings-path \
# search-index python /app/index_data.py --text-ids-path /text-ids-path --embeddings-path /embeddings-path --desc-ids-path /desc-ids-path --desc-embeddings-path /desc-embeddings-path -d 768

# OPENSEARCH_USER=cpr-master \
# OPENSEARCH_PASSWORD=fQ7c@888Etz@ \
# OPENSEARCH_URL=https://search-navigator-alpha-g5fgeoght3wpmpk2jjxopbaaue.eu-west-2.es.amazonaws.com \
# OPENSEARCH_INDEX=navigator \
# OPENSEARCH_REQUEST_TIMEOUT=30 \
# OPENSEARCH_USE_SSL=False \
# OPENSEARCH_VERIFY_CERTS=False \
# OPENSEARCH_SSL_WARNINGS=False \
# OPENSEARCH_INDEX_EMBEDDING_DIM=768 \
# BACKEND_DATABASE_URL=postgresql://navigator:password@localhost:5432/navigator \
# python search-index/index_data.py \
# --text-ids-path ~/Documents/CPR/data/_alpha-temp/final/ids_msmarco-distilbert-dot-v5_20220512-1415.json \
# --embeddings-path ~/Documents/CPR/data/_alpha-temp/final/embeddings_dim_768_msmarco-distilbert-dot-v5_20220512-1415.memmap \
# --desc-ids-path ~/Documents/CPR/data/_alpha-temp/final/description_ids_msmarco-distilbert-dot-v5_20220512-1415.csv \
# --desc-embeddings-path ~/Documents/CPR/data/_alpha-temp/final/description_embs_dim_768_msmarco-distilbert-dot-v5_20220512-1415.memmap \
# -d 768

OPENSEARCH_USER=admin \
OPENSEARCH_PASSWORD=admin \
OPENSEARCH_URL=http://localhost:9200 \
OPENSEARCH_INDEX=navigator \
OPENSEARCH_REQUEST_TIMEOUT=30 \
OPENSEARCH_USE_SSL=False \
OPENSEARCH_VERIFY_CERTS=False \
OPENSEARCH_SSL_WARNINGS=False \
OPENSEARCH_INDEX_EMBEDDING_DIM=768 \
BACKEND_DATABASE_URL=postgresql://navigator:password@localhost:5432/navigator \
python search-index/index_data.py \
--text-ids-path ~/Documents/CPR/data/_alpha-temp/test/ids_msmarco-distilbert-dot-v5_20220525-1506.json \
--embeddings-path ~/Documents/CPR/data/_alpha-temp/test/embeddings_dim_768_msmarco-distilbert-dot-v5_20220525-1506.memmap \
--desc-ids-path ~/Documents/CPR/data/_alpha-temp/test/description_ids_msmarco-distilbert-dot-v5_20220525-1506.csv \
--desc-embeddings-path ~/Documents/CPR/data/_alpha-temp/test/description_embs_dim_768_msmarco-distilbert-dot-v5_20220525-1506.memmap \
-d 768

# docker-compose run -v ~/Documents/CPR/data/search/embeddings/ids_msmarco-distilbert-dot-v5_20220420-1058.json:/text-ids-path \
# -v ~/Documents/CPR/data/search/embeddings/embeddings_dim_768_msmarco-distilbert-dot-v5_20220420-1058.memmap:/embeddings-path \
# -v ~/Documents/CPR/data/search/embeddings/description_ids_msmarco-distilbert-dot-v5_20220420-1058.csv:/desc-ids-path \
# -v ~/Documents/CPR/data/search/embeddings/description_embs_dim_768_msmarco-distilbert-dot-v5_20220420-1058.memmap:/desc-embeddings-path \
# search-index python /app/index_data.py --text-ids-path /text-ids-path --embeddings-path /embeddings-path --desc-ids-path /desc-ids-path --desc-embeddings-path /desc-embeddings-path 