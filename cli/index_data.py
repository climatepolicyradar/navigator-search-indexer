"""Index data into a running Opensearch index."""

import os
import sys
import time
import logging
import logging.config
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union, cast

import click
from cloudpathlib import S3Path
from tqdm.auto import tqdm
from cpr_data_access.parser_models import ParserOutput

from src.index.opensearch import populate_opensearch
from src.index.vespa_ import populate_vespa

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


def _get_index_tasks(
    text2embedding_output_dir: str,
    s3: bool,
    files_to_index: Optional[str] = None,
    limit: Optional[int] = None,
) -> Tuple[Sequence[ParserOutput], Union[Path, S3Path]]:
    if s3:
        embedding_dir_as_path = cast(S3Path, S3Path(text2embedding_output_dir))
    else:
        embedding_dir_as_path = Path(text2embedding_output_dir)

    _LOGGER.info(f"Getting tasks from {'s3' if s3 else 'local'}")
    tasks = [
        ParserOutput.model_validate_json(path.read_text())
        for path in tqdm(list(embedding_dir_as_path.glob("*.json")))
    ]

    if files_to_index is not None:
        tasks = [
            task for task in tasks if task.document_id in files_to_index.split(",")
        ]

        if missing_ids := set(files_to_index.split(",")) - set(
            [task.document_id for task in tasks]
        ):
            _LOGGER.warning(
                f"Missing files in the input directory for {', '.join(missing_ids)}"
            )

    if limit is not None:
        tasks = tasks[:limit]

    return tasks, embedding_dir_as_path


@click.command()
@click.argument("indexer_input_dir")
@click.option(
    "--s3",
    is_flag=True,
    required=False,
    help="Whether or not we are reading from and writing to S3.",
)
@click.option(
    "--files-to-index",
    required=False,
    help="Comma-separated list of IDs of files to index.",
)
@click.option(
    "--limit",
    "-l",
    type=int,
    required=False,
    help="Optionally limit the number of documents to index.",
)
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
    s3: bool,
    files_to_index: Optional[str],
    limit: Optional[int],
    index_type: str,
) -> None:
    if index_type.lower() == "opensearch":
        tasks, embedding_dir_as_path = _get_index_tasks(
            indexer_input_dir, s3, files_to_index, limit
        )
        populate_opensearch(tasks=tasks, embedding_dir_as_path=embedding_dir_as_path)
        sys.exit(0)
    elif index_type.lower() == "vespa":
        _LOGGER.warning("Vespa indexing still experimental")
        tasks, embedding_dir_as_path = _get_index_tasks(
            indexer_input_dir, s3, files_to_index, limit
        )
        start = time.time()
        populate_vespa(tasks=tasks, embedding_dir_as_path=embedding_dir_as_path)
        duration = time.time() - start
        _LOGGER.info(f"Vespa indexing completed after: {duration}s")
        sys.exit(0)
    _LOGGER.error(f"Unknown index type: {index_type}")
    sys.exit(1)


if __name__ == "__main__":
    run_as_cli()
