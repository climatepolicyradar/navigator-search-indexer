"""In-app config. Set by environment variables."""

import os
from typing import Set


class ConfigError(Exception):
    """Configuration Error."""

    pass


def _convert_to_bool(x: str) -> bool:
    if x.lower() == "true":
        return True
    elif x.lower() == "false":
        return False

    raise ValueError(f"Cannot convert {x} to bool. Input must be 'True' or 'False'.")


# General config
INDEX_ENCODER_CACHE_FOLDER: str = os.getenv("INDEX_ENCODER_CACHE_FOLDER", "/models")
ENCODING_BATCH_SIZE: int = int(os.getenv("ENCODING_BATCH_SIZE", "32"))
CDN_URL: str = os.getenv("CDN_URL", "https://cdn.climatepolicyradar.org")
TARGET_LANGUAGES: Set[str] = set(
    os.getenv("TARGET_LANGUAGES", "en").lower().split(",")
)  # comma-separated 2-letter ISO codes
FILES_TO_PROCESS = os.getenv("FILES_TO_PROCESS")
BLOCKS_TO_FILTER = os.getenv("BLOCKS_TO_FILTER", "Table,Figure").split(",")
ENCODER_SUPPORTED_LANGUAGES: Set[str] = {"en"}


# Opensearch Config
OPENSEARCH_INDEX_PREFIX: str = os.getenv("OPENSEARCH_INDEX_PREFIX", "navigator")
SBERT_MODEL: str = os.getenv("SBERT_MODEL", "msmarco-distilbert-dot-v5")
KNN_PARAM_EF_SEARCH: int = int(os.getenv("KNN_PARAM_EF_SEARCH", "100"))
OPENSEARCH_INDEX_NUM_SHARDS: int = int(os.getenv("OPENSEARCH_INDEX_NUM_SHARDS", "1"))
OPENSEARCH_INDEX_NUM_REPLICAS: int = int(
    os.getenv("OPENSEARCH_INDEX_NUM_REPLICAS", "2")
)
OPENSEARCH_INDEX_EMBEDDING_DIM: int = int(
    os.getenv("OPENSEARCH_INDEX_EMBEDDING_DIM", "768")
)
NMSLIB_EF_CONSTRUCTION: int = int(
    os.getenv("NMSLIB_EF_CONSTRUCTION", "512")
)  # TODO: tune me. 512 is Opensearch default
NMSLIB_M: int = int(
    os.getenv("NMSLIB_M", "16")
)  # TODO: tune me. 16 is Opensearch default
OPENSEARCH_USE_SSL: bool = _convert_to_bool(os.getenv("OPENSEARCH_USE_SSL", "True"))
OPENSEARCH_VERIFY_CERTS: bool = _convert_to_bool(
    os.getenv("OPENSEARCH_VERIFY_CERTS", "True")
)
OPENSEARCH_SSL_SHOW_WARN: bool = _convert_to_bool(
    os.getenv("OPENSEARCH_SSL_SHOW_WARN", "True")
)
OPENSEARCH_BULK_REQUEST_TIMEOUT: int = int(
    os.getenv("OPENSEARCH_BULK_REQUEST_TIMEOUT", "60")
)


# Vespa config
VESPA_DOCUMENT_BATCH_SIZE: int = int(os.getenv("VESPA_BATCH_SIZE", "2000"))
VESPA_INSTANCE_URL: str = os.getenv("VESPA_INSTANCE_URL", "")
VESPA_CERT_LOCATION: str = os.getenv("VESPA_CERT_LOCATION", "")
VESPA_KEY_LOCATION: str = os.getenv("VESPA_KEY_LOCATION", "")
VESPA_NAMESPACE_PREFIX: str = os.getenv("VESPA_NAMESPACE_PREFIX", "navigator")
DEV_MODE: bool = os.getenv("DEV_MODE", "False").lower() == "true"
