import os
import sys
import json
import time
import logging
import logging.config
from typing import Optional

import click

from cloudpathlib import S3Path

from src.index.vespa_ import populate_vespa, DocumentID

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
DEFAULT_LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",  # Default is stderr
            "formatter": "json",
        },
    },
    "loggers": {},
    "root": {
        "handlers": ["console"],
        "level": LOG_LEVEL,
    },
    "formatters": {"json": {"()": "pythonjsonlogger.jsonlogger.JsonFormatter"}},
}

_LOGGER = logging.getLogger(__name__)
logging.config.dictConfig(DEFAULT_LOGGING)

os.environ["CLOUPATHLIB_FILE_CACHE_MODE"] = "close_file"


@click.command()
@click.argument("indexer_input_dir")
@click.option(
    "--files-to-index",
    required=True,
    help="Comma-separated list of IDs of files to index.",
)
@click.option(
    "--limit",
    "-l",
    type=int,
    required=False,
    help="Optionally limit the number of documents to index.",
)
# TODO: Remove
@click.option(
    "--index-type",
    "-i",
    type=click.Choice(["opensearch", "vespa"]),
    default="opensearch",
    required=False,
    help="Which search database type to populate.",
)
def run_as_cli(
    indexer_input_dir: str,
    files_to_index: Optional[str],
    limit: Optional[int],
    index_type: str,
) -> None:
    if index_type.lower() == "opensearch":
        click.echo(f"Index type: {index_type}, is no longer used", err=True)
        sys.exit(1)
    elif index_type.lower() == "vespa":
        _LOGGER.warning("Vespa indexing still experimental")

        indexer_input_path = S3Path(indexer_input_dir)

        document_ids: list[DocumentID] = [
            DocumentID(doc_id) for doc_id in json.loads(files_to_index)
        ]
        document_s3_paths: list[S3Path] = [
            indexer_input_path / f"{document_id}.json" for document_id in document_ids
        ]

        start = time.time()
        # TODO: Confirm how we handle missing files.
        populate_vespa(
            paths=document_s3_paths, embedding_dir_as_path=indexer_input_path
        )
        duration = time.time() - start

        _LOGGER.info(f"Vespa indexing completed after: {duration}s")
        sys.exit(0)

    _LOGGER.error(f"Unknown index type: {index_type}")
    sys.exit(1)


if __name__ == "__main__":
    run_as_cli()
