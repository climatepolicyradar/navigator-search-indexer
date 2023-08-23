import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Union, List, Set, Sequence

import numpy as np

from src import config

from cpr_data_access.parser_models import ParserOutput, TextBlock, BlockType
from src.ml import SentenceEncoder
from src.s3 import get_s3_keys_with_prefix, s3_object_read_text

logger = logging.getLogger(__name__)


def replace_text_blocks(block: ParserOutput, new_text_blocks: Sequence[TextBlock]):
    """Updates the text blocks in the ParserOutput object."""
    if block.pdf_data is not None:
        block.pdf_data.text_blocks = new_text_blocks
    elif block.html_data is not None:
        block.html_data.text_blocks = new_text_blocks

    return block


def filter_blocks(
    parser_output: ParserOutput, remove_block_types: Sequence[str]
) -> Sequence[TextBlock]:
    """
    Given an ParserOutput filter the contained TextBlocks.

    Return this as a list of TextBlocks.
    """
    filtered_blocks = []
    for block in parser_output.get_text_blocks(including_invalid_html=True):
        if block.type.title() not in remove_block_types:
            filtered_blocks.append(block)
        else:
            logger.info(
                f"Filtered {block.type} block from {parser_output.document_id}.",
                extra={
                    "props": {
                        "document_id": parser_output.document_id,
                        "block_type": block.type,
                        "remove_block_types": remove_block_types,
                    }
                },
            )
    return filtered_blocks


def filter_on_block_type(
    inputs: Sequence[ParserOutput], remove_block_types: List[str]
) -> Sequence[ParserOutput]:
    """
    Filter a sequence of ParserOutputs.

    Remove the text blocks that are of the types declared in the remove block types
    array.
    """
    for _filter in remove_block_types:
        try:
            BlockType(_filter)
        except NameError:
            logger.warning(
                f"Blocks to filter should be of a known block type, removing {_filter} "
                f"from the list. "
            )
            remove_block_types.remove(_filter)

    return [
        replace_text_blocks(
            block=_input,
            new_text_blocks=filter_blocks(
                parser_output=_input, remove_block_types=remove_block_types
            ),
        )
        for _input in inputs
    ]


def get_ids_with_suffix(files: Sequence[str], suffix: str) -> Set[str]:
    """Get a set of the ids of the files with the given suffix."""
    files = [file for file in files if file.endswith(suffix)]
    return set([os.path.splitext(os.path.basename(file))[0] for file in files])


def encode_parser_output(
    encoder: SentenceEncoder,
    input_obj: ParserOutput,
    batch_size: int,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Encode a parser output object.

    Produce a numpy array of description embedding and a numpy array of text
    embeddings for a parser output.

    :param encoder: sentence encoder
    :param input_obj: parser output object
    :param batch_size: batch size for encoding text blocks
    :param device: device to use for encoding
    """

    description_embedding = encoder.encode(
        input_obj.document_description, device=device
    )

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


def get_files_to_process(
    s3: bool, input_dir: str, output_dir: str, redo: bool, limit: Union[None, int]
) -> Sequence[str]:
    """
    Get the list of files to process.

    Either from the config or from the input directory.
    """
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
            f"{len(document_ids_previously_parsed.intersection(files_to_process_ids))} "
            f"documents found that have already been encoded. Skipping. "
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
            f"Limiting to {files_to_process_ids} documents as the --limit flag has "
            f"been passed. "
        )
        files_to_process_ids = files_to_process_ids[:limit]

    return files_to_process_ids


def get_Text2EmbeddingsInput_array(
    input_dir: str, s3: bool, files_to_process_ids: Sequence[str]
) -> List[ParserOutput]:
    """Construct ParserOutput objects from parser output jsons.

    These objects will be used to generate embeddings and are either read in from S3
    or from the local file system.
    """
    return [
        ParserOutput.parse_raw(
            s3_object_read_text(os.path.join(input_dir, id_ + ".json"))
            if s3
            else Path(os.path.join(input_dir, id_ + ".json")).read_text()
        )
        for id_ in files_to_process_ids
    ]
