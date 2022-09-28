"""CLI to convert JSON documents outputted by the PDF parsing pipeline to embeddings."""

import logging
import logging.config
import os
from pathlib import Path
from typing import Optional, Tuple

import click
import numpy as np
from cloudpathlib import S3Path
from tqdm.auto import tqdm

from src.base import IndexerInput
from src.ml import SBERTEncoder, SentenceEncoder
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


def encode_indexer_input(
    encoder: SentenceEncoder, input: IndexerInput, batch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produce a numpy array of description embedding and a numpy array of text embeddings for an indexer input.

    :param input: serialised indexer input (output from document parser)
    :return: description embedding, text embeddings
    """

    description_embedding = encoder.encode(input.document_description)

    text_blocks = input.get_text_blocks()

    text_embeddings = encoder.encode_batch(
        [block.to_string() for block in text_blocks], batch_size=batch_size
    )

    return description_embedding, text_embeddings


@click.command()
@click.argument(
    "input-dir",
)
@click.argument(
    "output-dir",
)
@click.option(
    "--s3",
    is_flag=True,
    required=False,
    help="Whether or not we are reading from and writing to S3.",
)
@click.option(
    "--redo",
    "-r",
    help="Redo encoding for files that have already been parsed. By default, files with IDs that already exist in the output directory are skipped.",
    is_flag=True,
    default=False,
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Optionally limit the number of text samples to process. Useful for debugging.",
)
def main(
    input_dir: str,
    output_dir: str,
    s3: bool,
    redo: bool,
    limit: Optional[int],
):
    """
    Run CLI to produce embeddings from document parser JSON outputs. Each embeddings file is called {id}.json where {id} is the document ID of the input. Its first line is the description embedding and all other lines are embeddings of each of the text blocks in the document in order. Encoding will automatically run on the GPU if one is available.

    Args:
        input_dir: Directory containing JSON files
        output_dir: Directory to save embeddings to
        s3: Whether we are reading from and writing to S3.
        redo: Redo encoding for files that have already been parsed. By default, files with IDs that already exist in the output directory are skipped.
        limit (Optional[int]): Optionally limit the number of text samples to process. Useful for debugging.
    """

    if s3:
        input_dir_as_path = S3Path(input_dir)
        output_dir_as_path = S3Path(output_dir)
    else:
        input_dir_as_path = Path(input_dir)
        output_dir_as_path = Path(output_dir)

    document_ids_previously_parsed = set(
        [path.stem for path in output_dir_as_path.glob("*.npy")]
    )
    files_to_parse = list(input_dir_as_path.glob("*.json"))
    tasks = [IndexerInput.parse_raw(path.read_text()) for path in files_to_parse]

    if not redo and document_ids_previously_parsed.intersection(
        {task.document_id for task in tasks}
    ):
        logger.warning(
            f"Found {len(document_ids_previously_parsed.intersection({task.document_id for task in tasks}))} documents that have already been encoded. Skipping."
        )
        tasks = [
            task
            for task in tasks
            if task.document_id not in document_ids_previously_parsed
        ]

        if not tasks:
            logger.warning("No more documents to encode. Exiting.")
            return

    # Filter only to tasks that have one language and where the language is supported. These could either be translated or in the original language.
    if (
        unsupported_languages := config.TARGET_LANGUAGES
        - config.ENCODER_SUPPORTED_LANGUAGES
    ):
        logger.warning(
            f"The following languages have been requested for encoding but are not supported by the encoder: {unsupported_languages}. Only the following languages will be encoded: {config.ENCODER_SUPPORTED_LANGUAGES}."
        )

    tasks = [
        task
        for task in tasks
        if task.languages
        and (len(task.languages) == 1)
        and (
            task.languages[0]
            in config.ENCODER_SUPPORTED_LANGUAGES.union(config.TARGET_LANGUAGES)
        )
    ]

    # TODO: check we have all the files we need here i.e. (no ids * no languages)? Or do in the indexing step?

    if limit:
        logger.info(
            f"Limiting to {limit} documents as the --limit flag has been passed."
        )
        tasks = tasks[:limit]

    logger.info(f"Loading sentence-transformer model {config.SBERT_MODEL}")
    encoder = SBERTEncoder(config.SBERT_MODEL)

    logger.info(
        f"Encoding text from {len(files_to_parse)} documents in batches of {config.ENCODING_BATCH_SIZE}"
    )
    for task in tqdm(tasks, unit="docs"):
        description_embedding, text_embeddings = encode_indexer_input(
            encoder, task, config.ENCODING_BATCH_SIZE
        )

        embeddings_output_path = output_dir_as_path / f"{task.document_id}.npy"

        combined_embeddings = np.vstack([description_embedding, text_embeddings])

        task_output_path = output_dir_as_path / f"{task.document_id}.json"
        task_output_path.write_text(task.json())

        embeddings_output_path = output_dir_as_path / f"{task.document_id}.npy"
        with embeddings_output_path.open("wb") as f:
            np.save(f, combined_embeddings, allow_pickle=False)


if __name__ == "__main__":
    main()
