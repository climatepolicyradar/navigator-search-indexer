import datetime
from typing import List
import pytest

from src.base import IndexerInput, DocumentMetadata, HTMLData
from src.test.test_utils import get_text_block


@pytest.fixture
def test_indexer_input_array() -> List[IndexerInput]:
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
        ),
    ]


@pytest.fixture
def test_indexer_input_no_lang() -> List[IndexerInput]:
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
            languages=None,
            translated=False,
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
        )
    ]


@pytest.fixture
def test_indexer_input_no_source_url() -> List[IndexerInput]:
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
            document_source_url=None,
            document_cdn_object=None,
            document_md5_sum="test_md5_sum",
            languages=None,
            translated=False,
            document_slug="test_slug",
            document_content_type=None,
            html_data=None,
            pdf_data=None,
        )
    ]