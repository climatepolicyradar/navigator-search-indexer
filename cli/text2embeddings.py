"""CLI to convert JSON documents outputted by the PDF parsing pipeline to embeddings."""

import logging
import logging.config
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import click
import numpy as np
from tqdm.auto import tqdm

from src.base import Text2EmbeddingsInput
from src.ml import SBERTEncoder, SentenceEncoder
from src import config
from src.utils import (
    filter_on_block_type,
    get_s3_keys_with_prefix,
    s3_object_read_text,
    save_ndarray_to_s3_as_npy,
    check_file_exists_in_s3,
    write_json_to_s3,
    get_ids_with_suffix,
)

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
    input_obj: Text2EmbeddingsInput,
    batch_size: int,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Produce a numpy array of description embedding and a numpy array of text embeddings for an indexer input.

    :param encoder: sentence encoder
    :param input_obj: indexer input object
    :param batch_size: batch size for encoding text blocks
    :param device: device to use for encoding
    """

    description_embedding = encoder.encode(input_obj.document_description, device=device)

    text_blocks = input_obj.get_text_blocks()

    if text_blocks:
        text_embeddings = encoder.encode_batch(
            [block.to_string() for block in text_blocks],
            batch_size=batch_size,
            device=device,
        )
    else:
        text_embeddings = None

    return description_embedding, text_embeddings


def get_files_to_process(s3: bool, input_dir: str, output_dir: str, redo: bool, limit: Union[None, int]) -> list:
    """Get the list of files to process, either from the config or from the input directory."""
    if s3:
        document_paths_previously_parsed = get_s3_keys_with_prefix(output_dir)
    else:
        document_paths_previously_parsed = set(os.listdir(output_dir))

    document_ids_previously_parsed = get_ids_with_suffix(
        document_paths_previously_parsed, ".npy"
    )

    if config.FILES_TO_PROCESS is not None:
        files_to_process_subset = config.FILES_TO_PROCESS.split("$")[1:]
        files_to_process = [os.path.join(input_dir, f) for f in files_to_process_subset]
    else:
        if s3:
            files_to_process = get_s3_keys_with_prefix(input_dir)
        else:
            files_to_process = os.listdir(input_dir)
    files_to_process_ids = get_ids_with_suffix(files_to_process, ".json")

    if not redo and document_ids_previously_parsed.intersection(files_to_process_ids):
        logger.warning(
            f"Found {len(document_ids_previously_parsed.intersection(files_to_process_ids))} documents that have "
            f"already been encoded. Skipping. "
        )
        files_to_process_ids = [
            id_
            for id_ in files_to_process_ids
            if id_ not in document_ids_previously_parsed
        ]

        if not files_to_process_ids:
            logger.warning("No more documents to encode. Exiting.")
            return []

    if limit:
        logger.info(
            f"Limiting to {files_to_process_ids} documents as the --limit flag has been passed."
        )
        files_to_process_ids = files_to_process_ids[:limit]

    return files_to_process_ids


def validate_languages_decorator(func):
    """Validate that the languages requested for encoding are supported by the encoder."""

    def wrapper(*args, **kwargs):
        if (
                unsupported_languages := config.TARGET_LANGUAGES
                - config.ENCODER_SUPPORTED_LANGUAGES
        ):
            logger.warning(
                f"The following languages have been requested for encoding but are not supported by the encoder: "
                f"{unsupported_languages}. Only the following languages will be encoded: "
                f"{config.ENCODER_SUPPORTED_LANGUAGES}. "
            )
        return func(*args, **kwargs)

    return wrapper


@validate_languages_decorator
def get_docs_of_supported_language(tasks: list[Text2EmbeddingsInput]):
    """Filter out documents that don't meet language requirements.

    Persist documents with either:
     - one language where the language is in the target languages
     - no language and no content type.

    This assumes that the document name and description are in English.
    """
    return [
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
           or (
                not task.languages
                and task.html_data is None
                and task.pdf_data is None
           )
    ]


def get_Text2EmbeddingsInput_array(
        input_dir: str, s3: bool, files_to_process_ids
) -> list[Text2EmbeddingsInput]:
    """Construct Text2EmbeddingsInput objects from parser output jsons.

    These objects will be used to generate embeddings and are either read in from S3 or from the local file
    system.
    """
    return [
        Text2EmbeddingsInput.parse_raw(
            s3_object_read_text(os.path.join(input_dir, id_ + ".json"))
            if s3
            else Path(os.path.join(input_dir, id_ + ".json")).read_text()
        )
        for id_ in files_to_process_ids
    ]


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
    help="Redo encoding for files that have already been parsed. By default, files with IDs that already exist "
         "in the output directory are skipped.",
    is_flag=True,
    default=False,
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"]),
    help="Device to use for embeddings generation",
    required=True,
    default="cpu",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Optionally limit the number of text samples to process. Useful for debugging.",
)
def run_as_cli(
        input_dir: str,
        output_dir: str,
        s3: bool,
        redo: bool,
        device: str,
        limit: Optional[int],
):
    """
    Run CLI to produce embeddings from document parser JSON outputs. Each embeddings file is called {id}.json
    where {id} is the document ID of the input. Its first line is the description embedding and all other lines
    are embeddings of each of the text blocks in the document in order. Encoding will automatically run on the
    GPU if one is available.

    Args: input_dir: Directory containing JSON files output_dir: Directory to save embeddings to s3: Whether we
    are reading from and writing to S3. redo: Redo encoding for files that have already been parsed. By default,
    files with IDs that already exist in the output directory are skipped. limit (Optional[int]): Optionally
    limit the number of text samples to process. Useful for debugging. device (str): Device to use for
    embeddings generation. Must be either "cuda" or "cpu".
    """
    # FIXME: This solution assumes that we have a json document with language = en (supported target language)
    #  for every document in the parser output. This isn't very robust. This solution also requires passing
    #  every document into the embeddings stage so we are declaring tasks that are immediately dropped due to
    #  content. Filter only to tasks that have one language and where the language is supported. These could
    #  either be translated or in the original language.

    logger.info(
        f"Running text2embeddings",
        extra={
            "props": {
                "input_dir": input_dir,
                "output_dir": output_dir,
                "s3": s3,
                "redo": redo,
                "device": device,
                "limit": limit,
            }
        },
    )

    logger.info("Getting files to process.")
    files_to_process_ids = get_files_to_process(s3, input_dir, output_dir, redo, limit)

    logger.info("Constructing Text2EmbeddingsInput objects from parser output jsons.")
    tasks = get_Text2EmbeddingsInput_array(input_dir, s3, files_to_process_ids)

    logger.info(
        "Filtering tasks to those with supported languages.",
        extra={"props": {"target_languages": config.TARGET_LANGUAGES}}
    )
    tasks = get_docs_of_supported_language(tasks)

    logger.info(
        "Filtering unwanted text block types.",
        extra={"props": {"BLOCKS_TO_FILTER": config.BLOCKS_TO_FILTER}},
    )
    tasks = filter_on_block_type(
        inputs=tasks, remove_block_types=config.BLOCKS_TO_FILTER
    )

    logger.info(f"Loading sentence-transformer model {config.SBERT_MODEL}")
    encoder = SBERTEncoder(config.SBERT_MODEL)

    logger.info(
        "Encoding text from documents.",
        extra={"props": {"ENCODING_BATCH_SIZE": config.ENCODING_BATCH_SIZE, "tasks_number": len(tasks)}}
    )
    for task in tqdm(tasks, unit="docs"):
        task_output_path = os.path.join(output_dir, task.document_id + ".json")

        try:
            write_json_to_s3(task.json(), task_output_path) if s3 else Path(
                task_output_path
            ).write_text(task.json())
        except Exception as e:
            logger.info(
                "Failed to write embeddings data to s3.",
                extra={"props": {"task_output_path": task_output_path, "exception": e}},
            )

        embeddings_output_path = os.path.join(output_dir, task.document_id + ".npy")

        file_exists = (
            check_file_exists_in_s3(embeddings_output_path)
            if s3
            else os.path.exists(embeddings_output_path)
        )
        if file_exists:
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

        save_ndarray_to_s3_as_npy(
            combined_embeddings, embeddings_output_path
        ) if s3 else np.save(embeddings_output_path, combined_embeddings)


if __name__ == "__main__":
    run_as_cli()
