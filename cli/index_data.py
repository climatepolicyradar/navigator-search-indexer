"""Index data into a running Opensearch index."""

import os
from pathlib import Path
from typing import Generator, Sequence, Union
import logging
import logging.config

import numpy as np
import click
from cloudpathlib import S3Path

from src.index import OpenSearchIndex
from src.base import IndexerInput
from src.index_mapping import COMMON_FIELDS
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

logger = logging.getLogger(__name__)
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
    required_fields = [field for fields in COMMON_FIELDS.values() for field in fields]

    return {k: v for k, v in task_dict.items() if k in required_fields}


def get_document_generator(
    tasks: Sequence[IndexerInput], embedding_dir_as_path: Union[Path, S3Path]
) -> Generator[dict, None, None]:
    """
    Generator for documents to index. For each input document, an Opensearch document is created for each of its text blocks as well as its title and description.

    :param tasks: list of tasks from the document parser
    :param embedding_dir_as_path: directory containing embeddings .npy files. These are named with IDs corresponding to the IDs in the tasks.
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

        # Generate text block docs
        text_blocks = task.get_text_blocks()

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


@click.command()
@click.argument("text2embedding-output-dir")
@click.option(
    "--s3",
    is_flag=True,
    required=False,
    help="Whether or not we are reading from and writing to S3.",
)
def main(
    text2embedding_output_dir: str,
    s3: bool,
) -> None:
    """
    Index documents into Opensearch.

    :param pdf_parser_output_dir: directory or S3 folder containing output JSON files from the PDF parser.
    :param embedding_dir: directory or S3 folder containing embeddings from the text2embeddings CLI.
    """
    if s3:
        embedding_dir_as_path = S3Path(text2embedding_output_dir)
    else:
        embedding_dir_as_path = Path(text2embedding_output_dir)

    tasks = [
        IndexerInput.parse_raw(path.read_text())
        for path in list(embedding_dir_as_path.glob("*.json"))
    ]

    doc_generator = get_document_generator(tasks, embedding_dir_as_path)

    opensearch = OpenSearchIndex(
        url=os.environ["OPENSEARCH_URL"],
        username=os.environ["OPENSEARCH_USER"],
        password=os.environ["OPENSEARCH_PASSWORD"],
        index_name=os.environ["OPENSEARCH_INDEX"],
        opensearch_connector_kwargs={
            "use_ssl": config.OPENSEARCH_USE_SSL,
            "verify_certs": config.OPENSEARCH_VERIFY_CERTS,
            "ssl_show_warn": config.OPENSEARCH_SSL_SHOW_WARN,
        },
        embedding_dim=config.OPENSEARCH_INDEX_EMBEDDING_DIM,
    )
    opensearch.delete_and_create_index(n_replicas=config.OPENSEARCH_INDEX_NUM_REPLICAS)
    # We disable index refreshes during indexing to speed up the indexing process,
    # and to ensure only 1 segment is created per shard. This also speeds up KNN
    # queries and aggregations according to the Opensearch and Elasticsearch docs.
    opensearch.set_index_refresh_interval(-1, timeout=60)
    opensearch.bulk_index(actions=doc_generator)

    # TODO: we wrap this in a try/except block because for now because sometimes it times out, and we don't want the whole >1hr indexing process to fail if this happens. We should stop doing this if we ever care what the refresh interval is, i.e. when we plan on incrementally adding data to the index.
    try:
        # 1 second refresh interval is the Opensearch default
        opensearch.set_index_refresh_interval(1, timeout=60)
    except Exception as e:
        logger.info(f"Failed to set index refresh interval after indexing: {e}")

    opensearch.warmup_knn()


if __name__ == "__main__":
    main()
