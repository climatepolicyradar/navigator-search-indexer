import datetime

import pytest

from src.base import IndexerInput, DocumentMetadata, HTMLData, TextBlock
from src.utils import filter_on_block_type


def get_text_block(text_block_type: str) -> TextBlock:
    """Returns a TextBlock object with the given type."""
    return TextBlock(
        text=["test_text"],
        text_block_id="test_text_block_id",
        language="test_language",
        type=text_block_type,
        type_confidence=1.0,
        coords=[(0, 0), (0, 0), (0, 0), (0, 0)],
        page_number=0,
    )


@pytest.fixture
def test_indexer_input_array() -> list[IndexerInput]:
    """Test IndexerInput array with html containing various text block types."""
    return [
        IndexerInput(
            document_id="test_id",
            document_metadata=DocumentMetadata(
                publication_ts=datetime.datetime.now(),
                date="test_date",
                geography="test_geography",
                category="test_category",
                source="test_source",
                type="test_type",
                sectors=["test_sector"],
            ),
            document_name="test_name",
            document_description="test_description",
            document_source_url="https://www.google.com/path.html",
            document_cdn_object="test_cdn_object",
            document_md5_sum="test_md5_sum",
            languages=["test_language"],
            translated=True,
            document_slug="test_slug",
            document_content_type="text/html",
            html_data=HTMLData(
                has_valid_text=True,
                text_blocks=[
                    get_text_block("Table"),
                    get_text_block("Text"),
                    get_text_block("Text"),
                    get_text_block("Figure"),
                    get_text_block("Text"),
                    get_text_block("Random"),
                    get_text_block("Google Text Block"),
                ],
            ),
            pdf_data=None,
        ),
        IndexerInput(
            document_id="test_id",
            document_metadata=DocumentMetadata(
                publication_ts=datetime.datetime.now(),
                date="test_date",
                geography="test_geography",
                category="test_category",
                source="test_source",
                type="test_type",
                sectors=["test_sector"],
            ),
            document_name="test_name",
            document_description="test_description",
            document_source_url="https://www.google.com/path.html",
            document_cdn_object="test_cdn_object",
            document_md5_sum="test_md5_sum",
            languages=["test_language"],
            translated=True,
            document_slug="test_slug",
            document_content_type="text/html",
            html_data=HTMLData(
                has_valid_text=False,
                text_blocks=[
                    get_text_block("Table"),
                    get_text_block("Text"),
                    get_text_block("Google Text Block"),
                ],
            ),
            pdf_data=None,
        )
    ]


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
    assert test_indexer_input_array[1].get_text_blocks(including_invalid_html=True) is not []
    assert len(test_indexer_input_array[1].get_text_blocks(including_invalid_html=True)) == 3
