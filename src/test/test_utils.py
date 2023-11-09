from typing import Sequence

import numpy as np
from cpr_data_access.parser_models import BlockType, ParserOutput, PDFTextBlock

from src import config
from src.ml import SBERTEncoder
from src.utils import (
    filter_on_block_type,
    replace_text_blocks,
    filter_blocks,
    get_ids_with_suffix,
    encode_parser_output,
)


def test_filter_on_block_type(test_parser_output_array):
    """Tests that the filter_on_block_type function removes the correct text blocks."""

    filtered_inputs = filter_on_block_type(
        inputs=test_parser_output_array, remove_block_types=["Text", "Figure"]
    )

    assert filtered_inputs[0].html_data is not None
    assert len(filtered_inputs[0].html_data.text_blocks) == 2

    assert filtered_inputs[0].html_data.text_blocks[0].type == "Table"
    assert filtered_inputs[0].html_data.text_blocks[0].text == ["test_text"]

    assert filtered_inputs[0].html_data.text_blocks[1].type == "Google Text Block"
    assert filtered_inputs[0].html_data.text_blocks[1].text == ["test_text"]

    # Assert that we can filter on IndexerInputs that don't have valid text
    assert filtered_inputs[1].html_data is not None
    assert len(filtered_inputs[1].html_data.text_blocks) == 2

    assert filtered_inputs[1].html_data.text_blocks[0].type == "Table"
    assert filtered_inputs[1].html_data.text_blocks[0].text == ["test_text"]

    assert filtered_inputs[1].html_data.text_blocks[1].type == "Google Text Block"
    assert filtered_inputs[1].html_data.text_blocks[1].text == ["test_text"]


def test_has_valid_text_override(test_parser_output_array: Sequence[ParserOutput]):
    """
    Test that the get_text_blocks method provides the right response.

    Particularly when using the including_invalid_html parameter.
    """

    output = test_parser_output_array[1]
    assert output.get_text_blocks() == []

    text_blocks_include_invalid = output.get_text_blocks(including_invalid_html=True)
    assert text_blocks_include_invalid is not None
    assert len(text_blocks_include_invalid) == 3


def test_replace_text_blocks(test_pdf_file_json):
    """Tests that the replace_text_blocks function replaces the correct text blocks."""
    parser_output = ParserOutput.model_validate(test_pdf_file_json)

    updated_parser_output = replace_text_blocks(
        block=parser_output,
        new_text_blocks=[
            PDFTextBlock(
                text=["test_text_2"],
                text_block_id="test_text_block_id_2",
                language="test_language_2",
                type=BlockType.TEXT,
                type_confidence=1.0,
                coords=[(0, 0), (0, 0), (0, 0), (0, 0)],
                page_number=0,
            )
        ],
    )

    assert updated_parser_output.pdf_data is not None
    assert len(updated_parser_output.pdf_data.text_blocks) == 1
    assert updated_parser_output.pdf_data.text_blocks[0].text == ["test_text_2"]


def test_filter_blocks(test_pdf_file_json):
    """Tests that the filter_blocks function removes the correct text blocks."""
    parser_output = ParserOutput.model_validate(test_pdf_file_json)

    filtered_text_blocks = filter_blocks(
        parser_output=parser_output, remove_block_types=["Text"]
    )

    for block in filtered_text_blocks:
        assert block.type != "Text"

    assert len(filtered_text_blocks) > 0


def test_get_ids_with_suffix():
    """Tests that get_ids_with_suffix function returns the correct filtered ids."""
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

    test_pdf_file_json.update(
        {
            "document_metadata": {
                "name": "test_updated_pdf",
                "description": "test_pdf_updated_description",
                "import_id": "CCLW.executive.1003.0",
                "family_import_id": "CCLW.executive.1003.0",
                "family_slug": "slug_CCLW.executive.1003.0",
                "slug": "test_pdf",
                "source_url": "https://cdn.climatepolicyradar.org/EUR/2013/EUR-2013-01-01-Overview+of+CAP+Reform+2014-2020_6237180d8c443d72c06c9167019ca177.pdf",
                "download_url": "https://cdn.climatepolicyradar.org/EUR/2013/EUR-2013-01-01-Overview+of+CAP+Reform+2014-2020_6237180d8c443d72c06c9167019ca177.pdf",
                "languages": ["en"],
                "metadata": {
                    "test_key": "test_value",
                    "sectors": ["sector1", "sector2"],
                },
                "publication_ts": "2022-10-25 12:43:00.869045",
                "geography": "test_geo",
                "category": "test_category",
                "source": "test_source",
                "type": "test_type",
            },
        }
    )

    input_obj = ParserOutput.model_validate(test_pdf_file_json)

    description_embeddings, text_embeddings = encode_parser_output(
        encoder=encoder_obj, input_obj=input_obj, batch_size=32
    )

    assert isinstance(description_embeddings, np.ndarray)
    assert isinstance(text_embeddings, np.ndarray)


# TODO get_files_to_process
#   TODO local files, s3 files, environment variable files

# TODO get_Text2EmbeddingsInput_array
#   TODO needs s3 files, local files, of the form json IndexerInput objects
