import logging
from typing import Sequence, Any
from io import BytesIO
from pathlib import Path
import numpy as np


from cpr_data_access.parser_models import BlockType, ParserOutput, TextBlock

_LOGGER = logging.getLogger(__name__)


def replace_text_blocks(block: ParserOutput, new_text_blocks: list[TextBlock]):
    """Updates the text blocks in the IndexerInput object."""
    if block.pdf_data is not None:
        block.pdf_data.text_blocks = new_text_blocks  # type: ignore
    elif block.html_data is not None:
        block.html_data.text_blocks = new_text_blocks  # type: ignore

    return block


def filter_blocks(
    indexer_input: ParserOutput, remove_block_types: list[str]
) -> list[TextBlock]:
    """Filter the contained TextBlocks and return this as a list of TextBlocks."""
    filtered_blocks = []
    for block in indexer_input.get_text_blocks(including_invalid_html=True):
        if block.type.title() not in remove_block_types:
            filtered_blocks.append(block)
        else:
            _LOGGER.info(
                f"Filtered {block.type} block from {indexer_input.document_id}.",
                extra={
                    "props": {
                        "document_id": indexer_input.document_id,
                        "block_type": block.type,
                        "remove_block_types": remove_block_types,
                    }
                },
            )
    return filtered_blocks


def filter_on_block_type(
    inputs: Sequence[ParserOutput], remove_block_types: list[str]
) -> Sequence[ParserOutput]:
    """
    Filter a sequence of IndexerInputs to remove unwanted TextBlocks.

    Unwanted text block types are the types declared in the remove block types array.
    """
    for _filter in remove_block_types:
        try:
            BlockType(_filter)
        except NameError:
            _LOGGER.warning(
                "Blocks to filter should be of a known block type, "
                f"removing {_filter} from the list."
            )
            remove_block_types.remove(_filter)

    return [
        replace_text_blocks(
            block=_input,
            new_text_blocks=filter_blocks(
                indexer_input=_input, remove_block_types=remove_block_types
            ),
        )
        for _input in inputs
    ]


def read_npy_file(file_path: Path) -> Any:
    """Read an npy file."""
    with open(file_path, "rb") as task_array_file_like:
        return np.load(BytesIO(task_array_file_like.read()))
