from pathlib import Path
from typing import Optional, Sequence

import pytest
from cpr_data_access.parser_models import ParserOutput

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


def test_vespa_document_generator(
    test_input_dir: Path,
):
    """Test that the document generator returns documents in the correct format."""

    tasks = [
        ParserOutput.model_validate_json(path.read_text())
        for path in list(test_input_dir.glob("*.json"))
    ]

    # checking that we've picked up some tasks, otherwise the test is pointless
    # because the document generator will be empty
    assert len(tasks) > 0

    doc_generator = get_document_generator(
        tasks=tasks,
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
