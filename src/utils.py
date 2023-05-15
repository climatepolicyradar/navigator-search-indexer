import logging

from src.base import IndexerInput, TextBlock, BlockTypes

logger = logging.getLogger(__name__)


def replace_text_blocks(block: IndexerInput, new_text_blocks: list[TextBlock]):
    """Updates the text blocks in the IndexerInput object."""
    logger.debug(f"Replacing text blocks in {block.document_id}")

    if block.pdf_data is not None:
        block.pdf_data.text_blocks = new_text_blocks
    elif block.html_data is not None:
        block.html_data.text_blocks = new_text_blocks

    return block


def filter_on_block_type(inputs: list[IndexerInput], remove_block_types: list[str]) -> list[IndexerInput]:
    """Filter a sequence of IndexerInputs to remove the textblocks that are of the types declared in the remove block
    types array."""
    for _filter in remove_block_types:
        try:
            BlockTypes(_filter)
        except NameError:
            logger.warning(f"Blocks to filter should be of a known block type, removing {_filter} from the list.")
            remove_block_types.remove(_filter)

    logger.debug(f"Filtering on block types: {remove_block_types}")
    return [
        replace_text_blocks(
            block=_input,
            new_text_blocks=[
                block for block in _input.get_text_blocks()
                if block.type.title() not in remove_block_types
            ]
        )
        for _input in inputs
    ]
