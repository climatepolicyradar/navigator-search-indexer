import os
import json
import logging
import logging.config

import click

from cloudpathlib import S3Path

from src.index.vespa_ import populate_vespa, DocumentID
from src import config

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


def identify_document_path(
    document_id: DocumentID, input_path: S3Path, target_lang: str
) -> S3Path:
    """
    Identify the document path in the embeddings_input directory.

    This directory contains both non-translated and translated versions of a document.
    If a translated version exists for the target language this will be used.
    """
    translated_path = input_path / f"{document_id}_translated_{target_lang}.json"

    if translated_path.exists():
        return translated_path

    return input_path / f"{document_id}.json"


@click.command()
@click.argument("embeddings_input_dir")
@click.argument("inference_results_s3_path")
@click.option(
    "--files-to-index",
    required=True,
    help="JSON array of document IDs to index.",
)
def run_as_cli(
    embeddings_input_dir: str,
    inference_results_s3_path: str,
    files_to_index: str,
) -> None:

    embeddings_input_s3_path = S3Path(embeddings_input_dir)
    inference_results_s3_path = S3Path(inference_results_s3_path)
    document_ids: list[DocumentID] = [
        DocumentID(doc_id) for doc_id in json.loads(files_to_index)
    ]

    assert len(config.TARGET_LANGUAGES) == 1, "Must be one target language."
    target_lang = next(iter(config.TARGET_LANGUAGES))

    document_s3_paths: list[S3Path] = []
    for document_id in document_ids:
        s3_path: S3Path = identify_document_path(
            document_id, embeddings_input_s3_path, target_lang
        )
        document_s3_paths.append(s3_path)

    populate_vespa(
        paths=document_s3_paths,
        inference_results_s3_path=inference_results_s3_path,
    )


if __name__ == "__main__":
    run_as_cli()
