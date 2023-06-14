"""In-app config. Set by environment variables."""

import os
from typing import Set
import re

SBERT_MODEL: str = os.getenv("SBERT_MODEL", "msmarco-distilbert-dot-v5")
INDEX_ENCODER_CACHE_FOLDER: str = os.getenv("INDEX_ENCODER_CACHE_FOLDER", "/models")
ENCODING_BATCH_SIZE: int = int(os.getenv("ENCODING_BATCH_SIZE", "32"))
TARGET_LANGUAGES: Set[str] = set(
    os.getenv("TARGET_LANGUAGES", "en").lower().split(",")
)  # comma-separated 2-letter ISO codes
ENCODER_SUPPORTED_LANGUAGES: Set[str] = {"en"}
FILES_TO_PROCESS = os.getenv("FILES_TO_PROCESS")
BLOCKS_TO_FILTER = os.getenv("BLOCKS_TO_FILTER", "Table,Figure").split(",")
# FIXME make sure this aligns with the updates to the fanout etc.
ID_PATTERN = re.compile(r"^[a-zA-Z0-9]+\.[a-zA-Z0-9]+\.[a-zA-Z0-9]+\.[a-zA-Z0-9]+")
S3_PATTERN = re.compile(r"s3://(?P<bucket>[\w-]+)/(?P<prefix>.+)")
