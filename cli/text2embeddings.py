"""CLI to convert JSON documents outputted by the PDF parsing pipeline to embeddings."""

import logging
import logging.config
import os
from pathlib import Path
from typing import Optional, Tuple

import click
import numpy as np
from cloudpathlib.exceptions import OverwriteNewerCloudError
from cloudpathlib import S3Path
from tqdm.auto import tqdm

from src.base import Text2EmbeddingsInput
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
    encoder: SentenceEncoder,
    input: Text2EmbeddingsInput,
    batch_size: int,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Produce a numpy array of description embedding and a numpy array of text embeddings for an indexer input.

    :param input: serialised indexer input (output from document parser)
    :return: description embedding, text embeddings. Text embeddings are None if there were no text blocks to encode.
    """

    description_embedding = encoder.encode(input.document_description, device=device)

    text_blocks = input.get_text_blocks()

    if text_blocks:
        text_embeddings = encoder.encode_batch(
            [block.to_string() for block in text_blocks],
            batch_size=batch_size,
            device=device,
        )
    else:
        text_embeddings = None

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
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"]),
    help="Device to use for embeddings generation",
    required=True,
    default="cpu",
)
def main(
    input_dir: str,
    output_dir: str,
    s3: bool,
    redo: bool,
    limit: Optional[int],
    device: str,
):
    """
    Run CLI to produce embeddings from document parser JSON outputs. Each embeddings file is called {id}.json where {id} is the document ID of the input. Its first line is the description embedding and all other lines are embeddings of each of the text blocks in the document in order. Encoding will automatically run on the GPU if one is available.

    Args:
        input_dir: Directory containing JSON files
        output_dir: Directory to save embeddings to
        s3: Whether we are reading from and writing to S3.
        redo: Redo encoding for files that have already been parsed. By default, files with IDs that already exist in the output directory are skipped.
        limit (Optional[int]): Optionally limit the number of text samples to process. Useful for debugging.
        device (str): Device to use for embeddings generation. Must be either "cuda" or "cpu".
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

    if config.FILES_TO_PROCESS is not None:
        files_to_process_subset = config.FILES_TO_PROCESS.split("$")[1:]
        files_to_process = [input_dir_as_path / f for f in files_to_process_subset]
    else:
        files_to_process = list(input_dir_as_path.glob("*.json"))

    tasks = [
        Text2EmbeddingsInput.parse_raw(path.read_text()) for path in files_to_process
    ]

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

    # Encode documents either with one language where the lanugage is in the target languages, or with no language and no content type. This assumes that the document name and description are in English.
    tasks = [
        task
        for task in tasks
        if (
            task.languages
            and (len(task.languages) == 1)
            and (
                task.languages[0]
                in config.ENCODER_SUPPORTED_LANGUAGES.union(config.TARGET_LANGUAGES)
            )
        )
        or (not task.languages and task.html_data is None and task.pdf_data is None)
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
        f"Encoding text from {len(files_to_process)} documents in batches "
        f"of {config.ENCODING_BATCH_SIZE} text blocks."
    )
    for task in tqdm(tasks, unit="docs"):
        task_output_path = output_dir_as_path / f"{task.document_id}.json"

        try:
            task_output_path.write_text(task.json())
        except OverwriteNewerCloudError:
            # TODO: investigate why this happens, and why we're copying the input
            logger.info(
                f"Tried to write to {task_output_path}, received "
                "OverwriteNewerCloudError, assuming a newer task definition "
                "is the one we want, continuing to process."
            )

        embeddings_output_path = output_dir_as_path / f"{task.document_id}.npy"
        if embeddings_output_path.exists():
            logger.info(
                f"Embeddings output file '{embeddings_output_path}' already exists, "
                "skipping processing."
            )
            continue

        description_embedding, text_embeddings = encode_indexer_input(
            encoder, task, config.ENCODING_BATCH_SIZE, device=device
        )
        combined_embeddings = (
            np.vstack([description_embedding, text_embeddings])
            if text_embeddings is not None
            else description_embedding.reshape(1, -1)
        )
        with embeddings_output_path.open("wb") as f:
            np.save(f, combined_embeddings, allow_pickle=False)


if __name__ == "__main__":
    main()
