import os
import json
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
@click.argument("inference_results_s3_path")
@click.option(
    "--files-to-index",
    required=True,
    help="Comma-separated list of IDs of files to index.",
)
def run_as_cli(
    indexer_input_dir: str,
    inference_results_s3_path: str,
    files_to_index: Optional[str],
) -> None:
    _LOGGER.warning("Vespa indexing still experimental")

    indexer_input_s3_path = S3Path(indexer_input_dir)
    inference_results_s3_path = S3Path(inference_results_s3_path)
    document_ids: list[DocumentID] = [
        DocumentID(doc_id) for doc_id in json.loads(files_to_index)
    ]

    document_s3_paths: list[S3Path] = []
    for document_id in document_ids:
        s3_path: S3Path = indexer_input_s3_path / f"{document_id}.json"
        if not s3_path.exists():
            _LOGGER.warning(f"S3 Path does not exist: {s3_path}")
        else:
            document_s3_paths.append(s3_path)

    populate_vespa(
        paths=document_s3_paths,
        indexer_input_s3_path=indexer_input_s3_path,
        inference_results_s3_path=inference_results_s3_path,
    )


if __name__ == "__main__":
    run_as_cli()
