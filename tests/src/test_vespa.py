from pathlib import Path

import boto3
from cloudpathlib import S3Path
from moto import mock_aws
import pytest

from src.index.vespa_ import (
    DocumentID,
    PassageID,
    get_existing_passage_ids,
    get_passage_id,
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


@pytest.mark.usefixtures("cleanup_test_vespa_before", "cleanup_test_vespa_after")
def test_get_document_generator(test_vespa, s3_mock, family_document_ids):
    """Assert that the vespa document generator works as expected."""
    path = S3Path(s3_mock.path)

    generator = get_document_generator(test_vespa, path)

    # The third fixture doc (CCLW.document.i00001057.n0000) is Portuguese and is
    # filtered out by the generator's language check, so it is not indexed.
    UNSUPPORTED_LANGUAGE_DOC_ID = "CCLW.document.i00001057.n0000"
    EXPECTED_DOCUMENTS = 2
    EXPECTED_PASSAGES = 130

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

    # Documents belong to the specific families - every fixture doc has passages
    assert len(family_document_refs) == EXPECTED_PASSAGES
    assert len(set(family_document_refs)) == EXPECTED_DOCUMENTS

    for doc_id in family_document_ids:
        if doc_id == UNSUPPORTED_LANGUAGE_DOC_ID:
            assert doc_id not in ids
            continue
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


@pytest.mark.usefixtures("cleanup_test_vespa_before", "cleanup_test_vespa_after")
def test_get_document_generator__real_export_with_line_separator(
    test_vespa, s3_bucket_and_region
):
    """Validate the generator can handle unicode escape characters."""
    fixture_path = (
        Path(__file__).parent.parent
        / "fixtures"
        / "pipeline_documents_for_indexing_v1"
        / "data_0_0_4.jsonl"
    )

    with mock_aws():
        s3 = boto3.client("s3", region_name=s3_bucket_and_region["region"])
        bucket = s3_bucket_and_region["bucket"]
        s3.create_bucket(
            Bucket=bucket,
            CreateBucketConfiguration={
                "LocationConstraint": s3_bucket_and_region["region"],
            },
        )
        key = "pipeline_documents_for_indexing/export.jsonl"
        s3.put_object(Bucket=bucket, Key=key, Body=fixture_path.read_bytes())

        path = S3Path(f"s3://{bucket}/{key}")
        ids = [doc_id for _, doc_id, _ in get_document_generator(test_vespa, path)]

    assert "Sabin.document.131481.131485" in ids


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
