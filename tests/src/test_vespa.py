from cloudpathlib import S3Path
from cpr_sdk.models.search import Passage
from cpr_sdk.parser_models import BlockType, ParserOutput, PDFTextBlock
import numpy as np
import pytest

from src.index.vespa_ import (
    DocumentID,
    PassageID,
    TextBlockId,
    VespaConcept,
    build_vespa_family_document,
    build_vespa_document_passage,
    get_existing_passage_ids,
    get_passage_id,
    join_concepts,
    passage_ids_match,
    retrieve_inference_result,
    reshape_metadata,
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

from tests.conftest import FIXTURE_DIR, INFERENCE_RESULTS_DIR, get_parser_output


INFERENCE_RESULTS_FIXTURE = FIXTURE_DIR / "inference_results"


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({}, []),
        (
            {"topic": ["Adaptation"]},
            [VespaFamilyDocument.MetadataItem(name="topic", value="Adaptation")],
        ),
        (
            {"family.id": [192949]},
            [VespaFamilyDocument.MetadataItem(name="family.id", value="192949")],
        ),
        (
            {
                "hazard": [],
                "sector": [
                    "Adaptation",
                    "Economy-wide",
                    "Waste",
                    "Agriculture",
                    "Water",
                ],
            },
            [
                VespaFamilyDocument.MetadataItem(name="sector", value="Adaptation"),
                VespaFamilyDocument.MetadataItem(name="sector", value="Economy-wide"),
                VespaFamilyDocument.MetadataItem(name="sector", value="Waste"),
                VespaFamilyDocument.MetadataItem(name="sector", value="Agriculture"),
                VespaFamilyDocument.MetadataItem(name="sector", value="Water"),
            ],
        ),
        (None, None),
    ],
)
def test_reshape_metadata(metadata, expected):
    result = reshape_metadata(metadata)
    assert result == expected


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


def test_join_concepts():
    """Test join_concepts sets concepts on passage from inference result by text_block_id."""
    parser_output: ParserOutput = get_parser_output(1, 1)
    text_block: PDFTextBlock = parser_output.pdf_data.text_blocks[0]

    # Document Passage and Inference Result with Matching Passage Id's
    document_passage: VespaDocumentPassage = build_vespa_document_passage(
        family_document_id="doc.1.1",
        search_weights_ref="id:doc_search:weight::default",
        text_block=text_block,
        embedding=np.array([-0.11900115, 0.17448892]),
    )
    concepts: list[VespaConcept] = [
        VespaConcept.model_validate(
            {
                "name": "test_concept",
                "id": "c1",
                "parent_concepts": [],
                "parent_concept_ids_flat": "",
                "start": 0,
                "end": 10,
                "model": "test",
                "timestamp": "2024-01-01T00:00:00",
            }
        )
    ]
    inference_result: dict[TextBlockId, list[VespaConcept]] = {
        TextBlockId(document_passage.text_block_id): concepts
    }
    result = join_concepts(document_passage, inference_result)
    assert result.concepts is not None
    assert (
        result.concepts == inference_result[TextBlockId(document_passage.text_block_id)]
    )

    # Inference Result that Doesn't match
    document_passage.concepts = []
    inference_result = {TextBlockId("NON_EXISTENT.executive.1.1"): concepts}
    result_empty = join_concepts(document_passage, inference_result)
    assert result_empty.concepts == []


def test_retrieve_inference_result__returns_data(s3_mock, family_document_ids):
    """Test retrieve_inference_result reads and parses inference result from S3."""
    result = retrieve_inference_result(
        S3Path(s3_mock.inference_results_path), family_document_ids[0]
    )

    assert result is not None
    assert isinstance(result, dict)
    for text_block_id, concepts in result.items():
        assert isinstance(text_block_id, str)
        assert all(isinstance(c, Passage.Concept) for c in concepts)


def test_retrieve_inference_result__returns_none_when_missing(s3_mock):
    """Test retrieve_inference_result returns None when file does not exist."""
    result = retrieve_inference_result(
        S3Path(s3_mock.inference_results_path), "NON_EXISTENT.document.1.1"
    )
    assert result is None


@pytest.mark.usefixtures("cleanup_test_vespa_before", "cleanup_test_vespa_after")
def test_get_existing_passage_ids__new_doc(test_vespa):
    new_id = "CCLW.executive.10014.111"
    existing_ids = get_existing_passage_ids(vespa=test_vespa, family_doc_id=new_id)
    assert not existing_ids


@pytest.mark.usefixtures(
    "cleanup_test_vespa_before", "preload_fixtures", "cleanup_test_vespa_after"
)
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


def test_passage_ids_match():
    """Test passage_ids_match compares inference result keys with text block ids."""
    parser_output = get_parser_output(1, 1)
    parser_output.pdf_data.text_blocks = [
        PDFTextBlock(
            text=["test_text"],
            text_block_id="p_1_b_0",
            type=BlockType.TEXT,
            type_confidence=1.0,
            coords=[(0, 0), (0, 0), (0, 0), (0, 0)],
            page_number=1,
        ),
        PDFTextBlock(
            text=["test_text"],
            text_block_id="p_1_b_1",
            type=BlockType.TEXT,
            type_confidence=1.0,
            coords=[(0, 0), (0, 0), (0, 0), (0, 0)],
            page_number=1,
        ),
        PDFTextBlock(
            text=["test_text"],
            text_block_id="p_1_b_2",
            type=BlockType.TEXT,
            type_confidence=1.0,
            coords=[(0, 0), (0, 0), (0, 0), (0, 0)],
            page_number=1,
        ),
    ]
    text_blocks = parser_output.get_text_blocks()
    assert len(text_blocks) == 3

    block_ids = [tb.text_block_id for tb in text_blocks]

    # Match: inference keys equal text block ids
    inference_match = {TextBlockId(bid): [] for bid in block_ids}
    assert passage_ids_match(inference_match, text_blocks) is True

    # No match: different ids
    inference_mismatch = {TextBlockId("other_id"): []}
    assert passage_ids_match(inference_mismatch, text_blocks) is False

    # No match: inference has extra id
    inference_extra = {
        TextBlockId(block_ids[0]): [],
        TextBlockId(block_ids[1]): [],
        TextBlockId(block_ids[2]): [],
        TextBlockId("extra_id"): [],
    }
    assert passage_ids_match(inference_extra, text_blocks) is False

    # No match: inference missing one id
    inference_missing = {
        TextBlockId(block_ids[0]): [],
        TextBlockId(block_ids[1]): [],
    }
    assert passage_ids_match(inference_missing, text_blocks) is False

    # No match: inference missing id (empty)
    assert passage_ids_match({}, text_blocks) is False

    # Match: both empty
    assert passage_ids_match({}, []) is True


@pytest.mark.usefixtures("cleanup_test_vespa_before", "cleanup_test_vespa_after")
def test_get_document_generator(test_vespa):
    """Assert that the vespa document generator works as expected."""
    indexer_input_s3_path = FIXTURE_DIR / "s3_files"
    paths = [
        indexer_input_s3_path / "CCLW.executive.10002.4495.json",
        indexer_input_s3_path / "CCLW.executive.10014.4470.json",
        indexer_input_s3_path / "CCLW.document.i00000004.n0000.json",
    ]

    fixture_doc_ids = []
    fixture_text_blocks = []
    for path in paths:
        fixture_content = ParserOutput.model_validate_json(path.read_text())
        fixture_doc_ids.append(fixture_content.document_id)
        fixture_text_blocks.extend(fixture_content.text_blocks)

    generator = get_document_generator(
        test_vespa, paths, indexer_input_s3_path, INFERENCE_RESULTS_DIR
    )

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


@pytest.mark.parametrize(
    ("text_block_id", "passage_idx", "expected_suffix"),
    [
        # v2: valid UUID — used verbatim
        (
            "550e8400-e29b-41d4-a716-446655440000",
            0,
            "550e8400-e29b-41d4-a716-446655440000",
        ),
        # v1: legacy format — falls back to loop index
        ("p_1_b_0", 0, "0"),
        ("p_1_b_0", 3, "3"),
    ],
)
def test_get_passage_id(text_block_id, passage_idx, expected_suffix):
    doc_id = DocumentID("CCLW.executive.1.0")
    result = get_passage_id(doc_id, text_block_id, passage_idx)
    assert result == PassageID(f"{doc_id}.{expected_suffix}")
