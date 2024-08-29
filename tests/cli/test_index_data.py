from pathlib import Path

import pytest
from vespa.application import Vespa

from src.index.vespa_ import (
    _NAMESPACE,
    DOCUMENT_PASSAGE_SCHEMA,
    FAMILY_DOCUMENT_SCHEMA,
    SEARCH_WEIGHTS_SCHEMA,
    get_document_generator,
)


@pytest.fixture()
def test_input_dir() -> Path:
    return (Path(__file__).parent / "test_data" / "index_data_input").resolve()


def assert_expected_document_fields_are_present(doc):
    expected_fields = [
        "search_weights_ref",
        "family_name",
        "family_name_index",
        "family_description",
        "family_description_index",
        "family_description_embedding",
        "family_import_id",
        "family_slug",
        "family_publication_ts",
        "family_publication_year",
        "family_category",
        "family_geography",
        "family_source",
        "document_import_id",
        "document_slug",
        "document_title",
        "family_geographies",
        "corpus_import_id",
        "corpus_type_name",
        "collection_title",
        "collection_summary",
    ]
    for field in expected_fields:
        assert doc.get(field) is not None, f"{field} was None"


@pytest.mark.usefixtures("cleanup_test_vespa_before", "cleanup_test_vespa_after")
def test_vespa_document_generator(
    test_vespa: Vespa,
    test_input_dir: Path,
):
    """Test that the document generator returns documents in the correct format."""

    paths = list(test_input_dir.glob("*.json"))
    assert len(paths) > 0

    doc_generator = get_document_generator(
        vespa=test_vespa,
        paths=paths,
        embedding_dir_as_path=test_input_dir,
    )

    id_start_string = f"id:{_NAMESPACE}"
    search_weights_ref = None
    last_family_ref = None
    last_passage_ref = None
    for schema_type, idx, doc in doc_generator:
        if schema_type == SEARCH_WEIGHTS_SCHEMA:
            assert idx is not None
            assert search_weights_ref is None  # we should only get one of these
            search_weights_ref = f"{id_start_string}:{schema_type}::{idx}"
            continue

        if schema_type == FAMILY_DOCUMENT_SCHEMA:
            assert doc.get("search_weights_ref") is not None
            assert doc.get("search_weights_ref") == search_weights_ref
            last_family_ref = f"{id_start_string}:{schema_type}::{idx}"
            assert_expected_document_fields_are_present(doc)
            continue

        if schema_type == DOCUMENT_PASSAGE_SCHEMA:
            assert doc.get("search_weights_ref") is not None
            assert doc.get("search_weights_ref") == search_weights_ref
            assert last_family_ref is not None
            assert doc.get("family_document_ref") is not None
            assert doc.get("family_document_ref") == last_family_ref
            last_passage_ref = f"{id_start_string}:{schema_type}::{idx}"
            continue

        assert False, "Unknown schema type"

    # Make sure we've seen every type of doc expected
    assert search_weights_ref is not None
    assert last_family_ref is not None
    assert last_passage_ref is not None
