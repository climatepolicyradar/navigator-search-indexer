import logging
from typing import Any, Optional, Sequence, Tuple, Union, cast
from io import BytesIO
from pathlib import Path
import numpy as np

from cloudpathlib import S3Path
from tqdm.auto import tqdm
from cpr_data_access.parser_models import BlockType, ParserOutput, TextBlock

_LOGGER = logging.getLogger(__name__)


def build_indexer_input_path(indexer_input_dir: str, s3: bool) -> Union[S3Path, Path]:
    _LOGGER.info(f"Tasks will be retrieved from {'s3' if s3 else 'local'}: {indexer_input_dir}")
    if s3:
        indexer_input_path = cast(S3Path, S3Path(indexer_input_dir))
    else:
        indexer_input_path = Path(indexer_input_dir)
    return indexer_input_path


def get_index_paths(
    indexer_input_path: str,
    files_to_index: Optional[str] = None,
    limit: Optional[int] = None,
) -> Tuple[Sequence[ParserOutput]]:

    files_to_index = files_to_index.split(",") if files_to_index else []

    paths = []
    doc_ids = []
    for i, path in enumerate(tqdm(list(indexer_input_path.glob("*.json"))), 1):        
        doc_id = path.stem

        if doc_id in files_to_index:
            continue

        paths.append(path)
        doc_ids.append(doc_id)

        if limit and i == limit:
            break

    if missing_ids := set(files_to_index) - set(doc_ids):
        _LOGGER.warning(
            f"Missing files in the input directory for {', '.join(missing_ids)}"
        )
    return paths


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
    input: ParserOutput, remove_block_types: list[str]
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

    return replace_text_blocks(
            block=input,
            new_text_blocks=filter_blocks(
                indexer_input=input, remove_block_types=remove_block_types
            ),
        )


def read_npy_file(file_path: Path) -> Any:
    """Read an npy file."""
    return np.load(BytesIO(file_path.read_bytes()))
