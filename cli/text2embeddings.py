"""CLI to convert JSON documents outputted by the PDF parsing pipeline to embeddings."""

import logging
from pathlib import Path
from typing import Optional, Tuple

import click
import numpy as np
from cloudpathlib import S3Path
from tqdm.auto import tqdm

from src.base import IndexerInput
from src.ml import SBERTEncoder, SentenceEncoder
from src import config

logger = logging.getLogger(__name__)


def encode_indexer_input(
    encoder: SentenceEncoder, input: IndexerInput, batch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produce a numpy array of description embedding and a numpy array of text embeddings for an indexer input.

    :param input: serialised indexer input (output from document parser)
    :return: description embedding, text embeddings
    """

    description_embedding = encoder.encode(input.document_description)

    if input.content_type == "text/html":
        text_blocks = input.html_data.text_blocks  # type: ignore
    elif input.content_type == "application/pdf":
        text_blocks = input.pdf_data.text_blocks  # type: ignore
    else:
        raise ValueError(f"Unknown content type {input.content_type}")

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
    "--limit",
    type=int,
    default=None,
    help="Optionally limit the number of text samples to process. Useful for debugging.",
)
def run_cli(
    input_dir: str,
    output_dir: str,
    s3: bool,
    limit: Optional[int],
):
    """
    Run CLI to produce embeddings from pdf2text JSON outputs. Encoding will automatically run on the GPU if one is available.

    Args:
        input_dir: Directory containing JSON files
        output_dir: Directory to save embeddings and IDs to
        s3: Whether we are reading from and writing to S3.
        limit (Optional[int]): Optionally limit the number of text samples to process. Useful for debugging.
    """

    if s3:
        input_dir_as_path = S3Path(input_dir)
        output_dir_as_path = S3Path(output_dir)
    else:
        input_dir_as_path = Path(input_dir)
        output_dir_as_path = Path(output_dir)

    files_to_parse = list(input_dir_as_path.glob("*.json"))
    tasks = [IndexerInput.parse_raw(path.read_text()) for path in files_to_parse]

    if limit:
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

        description_embedding_output_path = (
            output_dir_as_path / f"{task.id}_description_embedding.npy"
        )
        text_embedding_output_path = (
            output_dir_as_path / f"{task.id}_text_embedding.npy"
        )

        with description_embedding_output_path.open("wb") as f:
            np.save(f, description_embedding, allow_pickle=False)

        with text_embedding_output_path.open("wb") as f:
            np.save(f, text_embeddings, allow_pickle=False)


if __name__ == "__main__":
    run_cli()
