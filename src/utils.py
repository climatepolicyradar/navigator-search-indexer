from src.base import IndexerInput


def filter_on_block_type(inputs: list[IndexerInput], remove_block_types: list[str]) -> list[IndexerInput]:
    """Filter a sequence of IndexerInputs to remove the textblocks that are of the type Table or Figure."""

    return [
        input.update_text_blocks(
            [
                block for block in input.get_text_blocks()
                if block.type not in remove_block_types
            ]
        )
        for input in inputs
    ]
