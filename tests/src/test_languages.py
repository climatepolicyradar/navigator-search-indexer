import pytest

from src.index.vespa_ import VespaFamilyDocument
from src.languages import doc_has_supported_language
from tests.conftest import get_pipeline_fixture_row


@pytest.mark.parametrize(
    ("document_id", "expected"),
    [
        # A single supported language.
        ("CCLW.legislative.8580.1568", True),  # ['English']
        # A single non-English *source* language. The content is translated to
        # English upstream, but the export records the original language, so the
        # supported-language check excludes it.
        ("CCLW.document.i00001057.n0000", False),  # ['Portuguese']
        # No language metadata at all, but a source url and passages - this
        # document cannot be classified, so it is not indexed.
        ("Sabin.document.133674.133677", False),  # []
        # Mixed languages - upstream cannot cleanly translate these, so their
        # original text reaches the indexer. Two entries naming the same language
        # at different ISO granularities count as mixed.
        ("CCLW.legislative.4777.1812", False),  # ['English', 'Portuguese']
        ("CCLW.document.i00004914.n0000", False),  # ['Nepali (...)', 'Nepali (...)']
    ],
)
def test_doc_has_supported_language(document_id, expected) -> None:
    """Tests that the function returns only docs of a supported language."""
    row = get_pipeline_fixture_row(document_id)
    document = VespaFamilyDocument.model_validate(row["vespa_family_document"])

    assert (
        doc_has_supported_language(document, row["vespa_document_passages"]) is expected
    )


def test_doc_has_supported_language__root_document() -> None:
    """A document with no source url, languages or passages is still indexable.

    Every row in the export has a source url and at least one passage, so this
    case is built from one rather than read straight out of the fixtures.
    """
    row = get_pipeline_fixture_row("CCLW.legislative.8580.1568")
    document = VespaFamilyDocument.model_validate(
        row["vespa_family_document"]
    ).model_copy(update={"document_languages": [], "document_source_url": None})

    assert doc_has_supported_language(document, []) is True
