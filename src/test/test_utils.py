import numpy as np

from src import config
from src.base import IndexerInput, TextBlock, Text2EmbeddingsInput
from src.ml import SBERTEncoder
from src.utils import (
    filter_on_block_type,
    replace_text_blocks,
    filter_blocks,
    get_ids_with_suffix,
    encode_indexer_input,
)
from cli.test.conftest import get_text_block, test_pdf_file_json


def test_filter_on_block_type(test_indexer_input_array):
    """Tests that the filter_on_block_type function removes the correct text blocks."""

    filtered_inputs = filter_on_block_type(
        inputs=test_indexer_input_array, remove_block_types=["Text", "Figure"]
    )

    assert len(filtered_inputs[0].html_data.text_blocks) == 3

    assert filtered_inputs[0].html_data.text_blocks[0].type == "Table"
    assert filtered_inputs[0].html_data.text_blocks[0].text == ["test_text"]

    assert filtered_inputs[0].html_data.text_blocks[1].type == "Random"
    assert filtered_inputs[0].html_data.text_blocks[1].text == ["test_text"]

    assert filtered_inputs[0].html_data.text_blocks[2].type == "Google Text Block"
    assert filtered_inputs[0].html_data.text_blocks[2].text == ["test_text"]

    # Assert that we can filter on IndexerInputs that don't have valid text
    assert len(filtered_inputs[1].html_data.text_blocks) == 2

    assert filtered_inputs[1].html_data.text_blocks[0].type == "Table"
    assert filtered_inputs[1].html_data.text_blocks[0].text == ["test_text"]

    assert filtered_inputs[1].html_data.text_blocks[1].type == "Google Text Block"
    assert filtered_inputs[1].html_data.text_blocks[1].text == ["test_text"]


def test_has_valid_text_override(test_indexer_input_array):
    """Test that the get_text_blocks method provides the right response when using the including_invalid_html
    parameter."""

    assert test_indexer_input_array[1].get_text_blocks() == []
    assert (
        test_indexer_input_array[1].get_text_blocks(including_invalid_html=True)
        is not []
    )
    assert (
        len(test_indexer_input_array[1].get_text_blocks(including_invalid_html=True))
        == 3
    )


def test_replace_text_blocks(test_pdf_file_json):
    """Tests that the replace_text_blocks function replaces the correct text blocks."""
    indexer_input = IndexerInput.parse_obj(test_pdf_file_json)

    updated_indexer_input = replace_text_blocks(
        block=indexer_input,
        new_text_blocks=[
            TextBlock(
                text=["test_text_2"],
                text_block_id="test_text_block_id_2",
                language="test_language_2",
                type="Text",
                type_confidence=1.0,
                coords=[(0, 0), (0, 0), (0, 0), (0, 0)],
                page_number=0,
            )
        ],
    )

    assert len(updated_indexer_input.pdf_data.text_blocks) == 1
    assert updated_indexer_input.pdf_data.text_blocks[0].text == ["test_text_2"]


def test_filter_blocks(test_pdf_file_json):
    """Tests that the filter_blocks function removes the correct text blocks."""
    indexer_input = IndexerInput.parse_obj(test_pdf_file_json)

    filtered_text_blocks = filter_blocks(
        indexer_input=indexer_input, remove_block_types=["Text"]
    )

    for block in filtered_text_blocks:
        assert block.type != "Text"

    assert len(filtered_text_blocks) > 0


def test_get_ids_with_suffix():
    """Tests that the get_ids_with_suffix function returns the correct ids after filtering."""
    filtered_ids = get_ids_with_suffix(
        files=[
            "s3://bucket/prefix/test_id_1.json",
            "s3://bucket/prefix/test_id_2.xlsx",
            "s3://bucket/prefix/test_id_3.npy",
            "s3://bucket/prefix/test_id_4.json",
        ],
        suffix=".json",
    )

    assert len(filtered_ids) == 2
    assert set(filtered_ids) == {"test_id_1", "test_id_4"}


def test_encode_indexer_input(test_pdf_file_json):
    """Tests that the encode_indexer_input function returns the correct embeddings."""
    encoder_obj = SBERTEncoder(config.SBERT_MODEL)

    test_pdf_file_json.update({"document_metadata": {"metadata_key": "metadata_value"}})

    input_obj = Text2EmbeddingsInput.parse_obj(test_pdf_file_json)

    description_embeddings, text_embeddings = encode_indexer_input(
        encoder=encoder_obj, input_obj=input_obj, batch_size=32
    )

    assert isinstance(description_embeddings, np.ndarray)
    assert isinstance(text_embeddings, np.ndarray)


# TODO get_files_to_process
#   TODO local files, s3 files, environment variable files

# TODO get_Text2EmbeddingsInput_array
#   TODO needs s3 files, local files, of the form json IndexerInput objects
