from cpr_data_access.parser_models import ParserOutput
import numpy as np
import pytest

from src.index.vespa_ import (
    build_vespa_family_document,
    build_vespa_document_passage,
    get_existing_passage_ids,
    remove_ids,
    determine_stray_ids,
    get_document_generator,
    VespaDocumentPassage,
    VespaFamilyDocument,
    VespaSearchWeights,
    SEARCH_WEIGHTS_SCHEMA,
    FAMILY_DOCUMENT_SCHEMA,
    DOCUMENT_PASSAGE_SCHEMA,
    _SCHEMAS_TO_PROCESS,
)

from tests.conftest import get_parser_output, FIXTURE_DIR


def test_build_vespa_family_document():
    parser_output = get_parser_output(1, 1)
    model = build_vespa_family_document(
        task=parser_output,
        embeddings=[np.array([-0.11900115, 0.17448892])],
        search_weights_ref="id:doc_search:weight::default",
    )
    VespaFamilyDocument.model_validate(model)


def test_build_vespa_document_passage():
    parser_output = get_parser_output(1, 1)
    text_block = parser_output.pdf_data.text_blocks[0]
    model = build_vespa_document_passage(
        family_document_id="doc.1.1",
        search_weights_ref="id:doc_search:weight::default",
        text_block=text_block,
        embedding=np.array([-0.11900115, 0.17448892]),
    )
    VespaDocumentPassage.model_validate(model)


@pytest.mark.usefixtures("cleanup_test_vespa_before", "cleanup_test_vespa_after")
def test_get_existing_passage_ids__new_doc(test_vespa):
    new_id = "CCLW.executive.10014.111"
    existing_ids = get_existing_passage_ids(vespa=test_vespa, family_doc_id=new_id)
    assert not existing_ids


@pytest.mark.usefixtures("cleanup_test_vespa_before", "cleanup_test_vespa_after")
def test_get_existing_passage_ids__existing_doc(test_vespa):
    family_doc_id = "CCLW.executive.10014.4470"
    start = get_existing_passage_ids(vespa=test_vespa, family_doc_id=family_doc_id)

    ids_to_remove = [
        "CCLW.executive.10014.4470.10",
        "CCLW.executive.10014.4470.13",
        "CCLW.executive.10014.4470.14",
        "CCLW.executive.10014.4470.16",
        "CCLW.executive.10014.4470.23",
        "CCLW.executive.10014.4470.26",
        "CCLW.executive.10014.4470.15",
        "CCLW.executive.10014.4470.2",
        "CCLW.executive.10014.4470.39",
    ]
    remove_ids(test_vespa, ids_to_remove)

    end = get_existing_passage_ids(vespa=test_vespa, family_doc_id=family_doc_id)

    assert len(end) == (len(start) - len(ids_to_remove))

    for i in ids_to_remove:
        assert i not in end


def test_determine_stray_ids():

    existing_doc_passage_ids = ["C.1.1", "C.1.2", "C.1.3", "C.1.4", "C.1.5"]
    new_passage_ids = ["C.1.1", "C.1.2", "C.1.3"]

    stray_ids = determine_stray_ids(
        existing_doc_passage_ids=existing_doc_passage_ids,
        new_passage_ids=new_passage_ids,
    )
    assert sorted(stray_ids) == ["C.1.4", "C.1.5"]


@pytest.mark.usefixtures("cleanup_test_vespa_before", "cleanup_test_vespa_after")
def test_get_document_generator(test_vespa):
    """Assert that the vespa document generator works as expected."""
    embedding_dir_as_path = FIXTURE_DIR / "s3_files"
    paths = [
        embedding_dir_as_path / "CCLW.executive.10002.4495.json",
        embedding_dir_as_path / "CCLW.executive.10014.4470.json",
        embedding_dir_as_path / "CCLW.document.i00000004.n0000.json",
    ]

    fixture_doc_ids = []
    fixture_text_blocks = []
    for path in paths:
        fixture_content = ParserOutput.model_validate_json(path.read_text())
        fixture_doc_ids.append(fixture_content.document_id)
        fixture_text_blocks.extend(fixture_content.text_blocks)

    generator = get_document_generator(test_vespa, paths, embedding_dir_as_path)

    EXPECTED_DOCUMENTS = 3
    EXPECTED_PASSAGES = 1978
    assert EXPECTED_DOCUMENTS == len(paths)
    assert EXPECTED_PASSAGES == len(fixture_text_blocks)


    schemas = []
    ids = []
    document_passage_ids = []
    family_document_refs = []
    for schema, doc_id, data in generator:
        schemas.append(schema)
        ids.append(doc_id)

        assert data
        assert isinstance(data, dict)

        if schema == SEARCH_WEIGHTS_SCHEMA:
            VespaSearchWeights.model_validate(data)
        elif schema == DOCUMENT_PASSAGE_SCHEMA:
            VespaDocumentPassage.model_validate(data)
            family_document_refs.append(data["family_document_ref"])
            document_passage_ids.append(doc_id)
        elif schema == FAMILY_DOCUMENT_SCHEMA:
            VespaFamilyDocument.model_validate(data)
        else:
            pytest.exit(f"Unexpected schema: {schema}")


    # Test schemas
    assert len(set(schemas)) == len(_SCHEMAS_TO_PROCESS)
    for schema in _SCHEMAS_TO_PROCESS:
        assert schema in schemas
    assert schemas.count("search_weights") == 1
    assert schemas.count("family_document") == EXPECTED_DOCUMENTS
    assert schemas.count("document_passage") == EXPECTED_PASSAGES

    # Test ids
    assert len(set(ids)) == len(ids)
    assert "default_weights" in ids

    # Documents belong to the specific families
    # We expect this to be 2 families as only two fixture docs have passages
    assert len(family_document_refs) == EXPECTED_PASSAGES
    assert len(set(family_document_refs)) == 2

    for doc_id in fixture_doc_ids:
        assert doc_id in ids
        assert doc_id not in document_passage_ids

    # Check every passage references a family document
    for ref in set(family_document_refs):
        # A document passage id CCLW.executive.0.0.0 would take the form
        # 'id:doc_search:family_document::CCLW.executive.0.0'
        id_parts = ref.split("::")
        family_schema = id_parts[0].split(":")[-1]
        family_id = id_parts[-1]

        assert family_schema == FAMILY_DOCUMENT_SCHEMA
        assert family_id in ids
        assert len(family_id.split(".")) == 4
