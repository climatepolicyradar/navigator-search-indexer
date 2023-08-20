"""Index data into a running Opensearch index."""

import os
import sys
import logging
import logging.config
from pathlib import Path
from typing import Generator, Optional, Sequence, Tuple, Union, cast

import numpy as np
import click
from cloudpathlib import S3Path
from tqdm.auto import tqdm

from src.index.opensearch import populate_opensearch
from src.base import IndexerInput, CONTENT_TYPE_HTML, CONTENT_TYPE_PDF
from src.index_mapping import COMMON_FIELDS
from src import config
from src.utils import filter_on_block_type

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


def get_metadata_dict(task: IndexerInput) -> dict:
    """
    Get key-value pairs for metadata fields: fields which are not required for search.

    :param task: task from the document parser
    :return dict: key-value pairs for metadata fields
    """

    task_dict = {
        **{k: v for k, v in task.dict().items() if k != "document_metadata"},
        **{f"document_{k}": v for k, v in task.document_metadata.dict().items()},
    }
    task_dict["document_name_and_slug"] = f"{task.document_name} {task.document_slug}"
    required_fields = [field for fields in COMMON_FIELDS.values() for field in fields]

    return {k: v for k, v in task_dict.items() if k in required_fields}


def get_core_document_generator(
    tasks: Sequence[IndexerInput], embedding_dir_as_path: Union[Path, S3Path]
) -> Generator[dict, None, None]:
    """
    Generator for core documents to index

    Documents to index are those with fields `for_search_document_name` and
    `for_search_document_description`.

    :param tasks: list of tasks from the document parser
    :param embedding_dir_as_path: directory containing embeddings .npy files.
        These are named with IDs corresponding to the IDs in the tasks.
    :yield Generator[dict, None, None]: generator of Opensearch documents
    """

    for task in tasks:
        all_metadata = get_metadata_dict(task)
        embeddings = np.load(str(embedding_dir_as_path / f"{task.document_id}.npy"))

        # Generate document name doc
        yield {
            **{"for_search_document_name": task.document_name},
            **all_metadata,
        }

        # Generate document description doc
        yield {
            **{"for_search_document_description": task.document_description},
            **all_metadata,
            **{"document_description_embedding": embeddings[0, :].tolist()},
        }


def get_text_document_generator(
    tasks: Sequence[IndexerInput],
    embedding_dir_as_path: Union[Path, S3Path],
    translated: Optional[bool] = None,
    content_types: Optional[Sequence[str]] = None,
) -> Generator[dict, None, None]:
    """
    Get generator for text documents to index.

    Documents to index are those containing text passages and their embeddings.
    Optionally filter by whether text passages have been translated and/or the
    document content type.

    :param tasks: list of tasks from the document parser
    :param embedding_dir_as_path: directory containing embeddings .npy files.
        These are named with IDs corresponding to the IDs in the tasks.
    :param translated: optionally filter on whether text passages are translated
    :param content_types: optionally filter on content types
    :yield Generator[dict, None, None]: generator of Opensearch documents
    """

    if translated is not None:
        tasks = [task for task in tasks if task.translated is translated]

    if content_types is not None:
        tasks = [task for task in tasks if task.document_content_type in content_types]

    _LOGGER.info(
        "Filtering unwanted text block types.",
        extra={"props": {"BLOCKS_TO_FILTER": config.BLOCKS_TO_FILTER}},
    )
    tasks = filter_on_block_type(
        inputs=tasks, remove_block_types=config.BLOCKS_TO_FILTER
    )

    for task in tasks:
        all_metadata = get_metadata_dict(task)
        embeddings = np.load(str(embedding_dir_as_path / f"{task.document_id}.npy"))

        # Generate text block docs
        text_blocks = task.vertically_flip_text_block_coords().get_text_blocks()

        for text_block, embedding in zip(text_blocks, embeddings[1:, :]):
            yield {
                **{
                    "text_block_id": text_block.text_block_id,
                    "text": text_block.to_string(),
                    "text_embedding": embedding.tolist(),
                    "text_block_coords": text_block.coords,
                    "text_block_page": text_block.page_number,
                },
                **all_metadata,
            }


def _get_index_tasks(
    text2embedding_output_dir: str,
    s3: bool,
    files_to_index: Optional[str] = None,
    limit: Optional[int] = None,
) -> Tuple[Sequence[IndexerInput], Union[Path, S3Path]]:
    if s3:
        embedding_dir_as_path = cast(S3Path, S3Path(text2embedding_output_dir))
    else:
        embedding_dir_as_path = Path(text2embedding_output_dir)

    _LOGGER.info(f"Getting tasks from {'s3' if s3 else 'local'}")
    tasks = [
        IndexerInput.parse_raw(path.read_text())
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


def main_opensearch(
    text2embedding_output_dir: str,
    s3: bool,
    files_to_index: Optional[str] = None,
    limit: Optional[int] = None,
) -> None:
    """
    Index documents into Opensearch.

    :param pdf_parser_output_dir: directory or S3 folder containing output JSON
        files from the PDF parser.
    :param embedding_dir: directory or S3 folder containing embeddings from the
        text2embeddings CLI.
    """
    tasks, embedding_dir_as_path = _get_index_tasks(
        text2embedding_output_dir, s3, files_to_index, limit
    )

    indices_to_populate: Sequence[Tuple[str, Generator[dict, None, None]]] = [
        (
            f"{config.OPENSEARCH_INDEX_PREFIX}_core",
            get_core_document_generator(tasks, embedding_dir_as_path),
        ),
        (
            f"{config.OPENSEARCH_INDEX_PREFIX}_pdfs_non_translated",
            get_text_document_generator(
                tasks,
                embedding_dir_as_path,
                translated=False,
                content_types=[CONTENT_TYPE_PDF],
            ),
        ),
        (
            f"{config.OPENSEARCH_INDEX_PREFIX}_pdfs_translated",
            get_text_document_generator(
                tasks,
                embedding_dir_as_path,
                translated=True,
                content_types=[CONTENT_TYPE_PDF],
            ),
        ),
        (
            f"{config.OPENSEARCH_INDEX_PREFIX}_htmls_non_translated",
            get_text_document_generator(
                tasks,
                embedding_dir_as_path,
                translated=False,
                content_types=[CONTENT_TYPE_HTML],
            ),
        ),
        (
            f"{config.OPENSEARCH_INDEX_PREFIX}_htmls_translated",
            get_text_document_generator(
                tasks,
                embedding_dir_as_path,
                translated=True,
                content_types=[CONTENT_TYPE_HTML],
            ),
        ),
    ]

    populate_opensearch(indices_to_populate)


@click.command()
@click.argument("text2embedding-output-dir")
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
    case_sensitive=False,
)
def run_as_cli(
    text2embedding_output_dir: str,
    s3: bool,
    files_to_index: Optional[str],
    limit: Optional[int],
    index_type: str,
) -> None:
    if index_type.lower() == "opensearch":
        main_opensearch(text2embedding_output_dir, s3, files_to_index, limit)
        sys.exit(0)
    if index_type.lower() == "vespa":
        # TODO: implement main_vespa(...)
        _LOGGER.error("Vespa indexing not yet implemented")
        sys.exit(1)
    _LOGGER.error(f"Unknown index type: {index_type}")
    sys.exit(1)


if __name__ == "__main__":
    run_as_cli()
