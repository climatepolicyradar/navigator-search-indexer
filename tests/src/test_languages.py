from datetime import datetime
from pydantic import AnyHttpUrl

from cpr_sdk.parser_models import (
    BackendDocument,
    BlockType,
    HTMLData,
    HTMLTextBlock,
    ParserOutput,
)

from src.languages import doc_has_supported_language

# TODO test that the warning is logged if the document language is not supported by
#  the encoder


def test_doc_has_supported_language() -> None:
    """Tests that the function returns only docs of a supported language."""
    metadata = BackendDocument(
        publication_ts=datetime(2013, 1, 1),
        name="Dummy Name",
        description="description",
        source_url="http://existing.com",
        type="EU Decision",
        source="CCLW",
        import_id="TESTCCLW.executive.4.4",
        family_import_id="TESTCCLW.family.4.0",
        family_slug="slug_TESTCCLW.family.4.0",
        category="Law",
        geography="EUR",
        languages=["English"],
        metadata={
            "hazards": [],
            "frameworks": [],
            "instruments": ["Capacity building|Governance"],
            "keywords": ["Adaptation"],
            "sectors": ["Economy-wide"],
            "topics": ["Adaptation"],
        },
        slug="dummy_slug",
    )

    html_blocks = [
        HTMLTextBlock(
            text=["test_text"],
            text_block_id="test_text_block_id",
            language="test_language",
            type=BlockType("Table"),
            type_confidence=1.0,
        ),
        HTMLTextBlock(
            text=["test_text"],
            text_block_id="test_text_block_id",
            language="test_language",
            type=BlockType("Google Text Block"),
            type_confidence=1.0,
        ),
    ]

    no_source_url_no_lang_no_data = ParserOutput(
        document_id="test_id",
        document_metadata=metadata,
        document_name="test_name",
        document_description="test_description",
        document_source_url=None,
        document_cdn_object="test_cdn_object",
        document_md5_sum="test_md5_sum",
        languages=None,
        translated=False,
        document_slug="test_slug",
        document_content_type=None,
        html_data=None,
        pdf_data=None,
    )

    source_url_no_lang_no_data = ParserOutput(
        document_id="test_id",
        document_metadata=metadata,
        document_name="test_name",
        document_description="test_description",
        document_source_url=AnyHttpUrl(
            "https://www.example.com/files/climate-document.pdf"
        ),
        document_cdn_object="test_cdn_object",
        document_md5_sum="test_md5_sum",
        languages=None,
        translated=False,
        document_slug="test_slug",
        document_content_type=None,
        html_data=None,
        pdf_data=None,
    )

    source_url_supported_lang_data = ParserOutput(
        document_id="test_id",
        document_metadata=metadata,
        document_name="test_name",
        document_description="test_description",
        document_source_url=AnyHttpUrl(
            "https://www.example.com/files/climate-document.pdf"
        ),
        document_cdn_object="test_cdn_object",
        document_md5_sum="test_md5_sum",
        languages=["en"],
        translated=False,
        document_slug="test_slug",
        document_content_type="text/html",
        html_data=HTMLData(has_valid_text=True, text_blocks=html_blocks),
        pdf_data=None,
    )

    source_url_unsupported_lang_data = ParserOutput(
        document_id="test_id",
        document_metadata=metadata,
        document_name="test_name",
        document_description="test_description",
        document_source_url=AnyHttpUrl(
            "https://www.example.com/files/climate-document.pdf"
        ),
        document_cdn_object="test_cdn_object",
        document_md5_sum="test_md5_sum",
        languages=["fr"],
        translated=False,
        document_slug="test_slug",
        document_content_type="text/html",
        html_data=HTMLData(has_valid_text=True, text_blocks=html_blocks),
        pdf_data=None,
    )

    assert doc_has_supported_language(no_source_url_no_lang_no_data) is True
    assert doc_has_supported_language(source_url_no_lang_no_data) is False
    assert doc_has_supported_language(source_url_supported_lang_data) is True
    assert doc_has_supported_language(source_url_unsupported_lang_data) is False
