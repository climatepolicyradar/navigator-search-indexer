import datetime
from pathlib import Path

from cloudpathlib import S3Path
import pytest
from pydantic import AnyHttpUrl

from cpr_data_access.parser_models import (
    BlockType,
    ParserOutput,
    BackendDocument,
    HTMLData,
    PDFTextBlock,
    HTMLTextBlock,
)

from src.utils import (
    build_indexer_input_path,
    filter_on_block_type,
    get_index_paths,
    parse_files_to_index,
)
from tests.conftest import FIXTURE_DIR


@pytest.mark.parametrize(
        "value, want",
        [
            (None, []),
            ("[]", []),
            ('["doc.1", "doc.2"]', ["doc.1","doc.2"]),
        ]
)
def test_parse_files_to_index(value, want):
    got = parse_files_to_index(value)
    assert got == want, f"Expected {want}, got {got}"


@pytest.mark.parametrize(
    "dir, use_s3, want",
    [
        ("local/path", False, Path("local/path")),
        ("s3://bucket/path", True, S3Path("s3://bucket/path")),
    ]
)
def test_build_indexer_input_path(dir, use_s3, want):
    got = build_indexer_input_path(dir, use_s3)
    assert got == want


@pytest.mark.parametrize(
    "files, limit, count",
    [
        (None, None, 3),
        (None, 1, 1),
        ('["CCLW.executive.10014.4470"]', None, 1),
    ]
)
def test_get_index_paths(files, limit, count):
    path = FIXTURE_DIR / "s3_files"
    got = get_index_paths(path, files, limit)
    assert len(got) == count
    for f in got:
        assert type(f) == type(path)


def get_pdf_text_block(text_block_type: str) -> PDFTextBlock:
    """Returns a PDFTextBlock object with the given type."""
    return PDFTextBlock(
        text=["test_text"],
        text_block_id="test_text_block_id",
        language="test_language",
        type=BlockType(text_block_type),
        type_confidence=1.0,
        coords=[(0, 0), (0, 0), (0, 0), (0, 0)],
        page_number=0,
    )


def get_html_text_block(text_block_type: str) -> HTMLTextBlock:
    """Returns a HMTMLTextBlock object with the given type."""
    return HTMLTextBlock(
        text=["test_text"],
        text_block_id="test_text_block_id",
        language="test_language",
        type=BlockType(text_block_type),
        type_confidence=1.0,
    )


@pytest.fixture
def test_indexer_input_array() -> list[ParserOutput]:
    """Test ParserOutput array with html containing various text block types."""
    return [
        ParserOutput(
            document_id="test_id",
            document_metadata=BackendDocument(
                name="test_name",
                description="test_description",
                import_id="test_id",
                slug="test_name_slug",
                family_import_id="test_family_id",
                family_slug="test_family_slug",
                publication_ts=datetime.datetime.now(),
                date="test_date",
                source_url=None,
                download_url=None,
                geography="test_geography",
                category="test_category",
                source="test_source",
                type="test_type",
                metadata={"sectors": ["test_sector"]},
                languages=[],
            ),
            document_name="test_name",
            document_description="test_description",
            document_source_url=AnyHttpUrl("https://www.google.com/path.html"),
            document_cdn_object="test_cdn_object",
            document_md5_sum="test_md5_sum",
            languages=["test_language"],
            translated=True,
            document_slug="test_slug",
            document_content_type="text/html",
            html_data=HTMLData(
                has_valid_text=True,
                text_blocks=[
                    get_html_text_block("Table"),
                    get_html_text_block("Text"),
                    get_html_text_block("Text"),
                    get_html_text_block("Figure"),
                    get_html_text_block("Text"),
                    get_html_text_block("Ambiguous"),
                    get_html_text_block("Google Text Block"),
                ],
            ),
            pdf_data=None,
        ),
        ParserOutput(
            document_id="test_id",
            document_metadata=BackendDocument(
                name="test_name",
                description="test_description",
                import_id="test_id",
                slug="test_name_slug",
                family_import_id="test_family_id",
                family_slug="test_family_slug",
                publication_ts=datetime.datetime.now(),
                date="test_date",
                source_url=None,
                download_url=None,
                geography="test_geography",
                category="test_category",
                source="test_source",
                type="test_type",
                metadata={"sectors": ["test_sector"]},
                languages=[],
            ),
            document_name="test_name",
            document_description="test_description",
            document_source_url=AnyHttpUrl("https://www.google.com/path.html"),
            document_cdn_object="test_cdn_object",
            document_md5_sum="test_md5_sum",
            languages=["test_language"],
            translated=True,
            document_slug="test_slug",
            document_content_type="text/html",
            html_data=HTMLData(
                has_valid_text=False,
                text_blocks=[
                    get_html_text_block("Table"),
                    get_html_text_block("Text"),
                    get_html_text_block("Google Text Block"),
                ],
            ),
            pdf_data=None,
        ),
    ]


def test_filter_on_block_type(test_indexer_input_array):
    """Tests that the filter_on_block_type function removes the correct text blocks."""

    filtered_input = filter_on_block_type(
        input=test_indexer_input_array[0], remove_block_types=["Text", "Figure"]
    )
    assert filtered_input.html_data is not None

    assert len(filtered_input.html_data.text_blocks) == 3

    assert filtered_input.html_data.text_blocks[0].type == "Table"
    assert filtered_input.html_data.text_blocks[0].text == ["test_text"]

    assert filtered_input.html_data.text_blocks[1].type == "Ambiguous"
    assert filtered_input.html_data.text_blocks[1].text == ["test_text"]

    assert filtered_input.html_data.text_blocks[2].type == "Google Text Block"
    assert filtered_input.html_data.text_blocks[2].text == ["test_text"]

    # Assert that we can filter on ParserOutputs that don't have valid text
    filtered_input = filter_on_block_type(
        input=test_indexer_input_array[1], remove_block_types=["Text", "Figure"]
    )
    assert filtered_input.html_data is not None
    assert len(filtered_input.html_data.text_blocks) == 2

    assert filtered_input.html_data.text_blocks[0].type == "Table"
    assert filtered_input.html_data.text_blocks[0].text == ["test_text"]

    assert filtered_input.html_data.text_blocks[1].type == "Google Text Block"
    assert filtered_input.html_data.text_blocks[1].text == ["test_text"]


def test_has_valid_text_override(test_indexer_input_array):
    """
    Test that the get_text_blocks method provides the right response.

    Tested when using the including_invalid_html parameter.
    """

    assert test_indexer_input_array[1].get_text_blocks() == []
    assert (
        test_indexer_input_array[1].get_text_blocks(including_invalid_html=True)
        is not []
    )
    assert (
        len(test_indexer_input_array[1].get_text_blocks(including_invalid_html=True))
        == 3
    )
