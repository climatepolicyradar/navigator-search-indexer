from unittest.mock import patch
import json
import traceback
import time
import uuid_utils as uuid

from click.testing import CliRunner
from cpr_sdk.parser_models import ParserOutput
import numpy as np
import pytest
from vespa.application import Vespa
from vespa.io import VespaQueryResponse
from typing import Any

from cli.index_data import run_as_cli
from conftest import FIXTURE_DIR, VESPA_TEST_ENDPOINT
from src import config
from src.index.vespa_ import (
    SEARCH_WEIGHTS_SCHEMA,
    FAMILY_DOCUMENT_SCHEMA,
    _NAMESPACE,
    DocumentID,
    get_existing_passage_ids,
)


def get_vespa_data(test_vespa: Vespa, schema: str, data_id: str):
    try:
        response = test_vespa.get_data(
            schema=schema, data_id=data_id, namespace=_NAMESPACE
        )
        assert response.status_code == 200
        return response.json
    except Exception as e:
        pytest.exit(e)


@patch.object(config, "VESPA_INSTANCE_URL", new=VESPA_TEST_ENDPOINT)
@patch.object(config, "DEVELOPMENT_MODE", new="true")
@pytest.mark.usefixtures("cleanup_test_vespa_before", "cleanup_test_vespa_after")
def test_integration(test_vespa, s3_mock, family_document_ids):
    """Run a single indexing and ensure all fields are populated"""
    runner = CliRunner()

    result = runner.invoke(
        run_as_cli,
        args=[
            s3_mock.path,
            s3_mock.inference_results_path,
            "--files-to-index",
            json.dumps(family_document_ids),
        ],
    )
    assert result.exit_code == 0, (
        f"Exception: {result.exception if result.exception else None}\n"
        f"Stdout: {result.stdout}"
        f"Trace: {traceback.print_exception(*result.exc_info)}"
    )

    for doc_id in family_document_ids:
        vespa_data = get_vespa_data(test_vespa, FAMILY_DOCUMENT_SCHEMA, doc_id)
        fixture_path = FIXTURE_DIR / "s3_files" / f"{doc_id}.json"
        embeddings_path = FIXTURE_DIR / "s3_files" / f"{doc_id}.npy"
        s3_data = ParserOutput.model_validate_json(fixture_path.read_text())
        embeddings = np.load(embeddings_path)

        vf = vespa_data["fields"]
        assert vf["family_name"] == s3_data.document_name
        assert vf["family_name_index"] == s3_data.document_name
        assert vf["family_description"] == s3_data.document_description
        assert vf["family_description_index"] == s3_data.document_description
        assert vf["family_description_embedding"]["values"] == embeddings[0].tolist()
        assert vf["family_import_id"] == s3_data.document_metadata.family_import_id
        assert vf["family_slug"] == s3_data.document_metadata.family_slug
        assert (
            vf["family_publication_ts"]
            == s3_data.document_metadata.publication_ts.isoformat()
        )
        assert (
            vf["family_publication_year"]
            == s3_data.document_metadata.publication_ts.year
        )
        assert vf["family_category"] == s3_data.document_metadata.category
        assert vf["family_geography"] == s3_data.document_metadata.geography
        assert vf["family_source"] == s3_data.document_metadata.source
        assert vf["document_import_id"] == s3_data.document_id
        assert vf["document_slug"] == s3_data.document_slug
        assert vf["document_languages"] == s3_data.document_metadata.languages
        assert vf["document_content_type"] == s3_data.document_content_type
        assert vf["document_md5_sum"] == s3_data.document_md5_sum
        assert vf["document_cdn_object"] == s3_data.document_cdn_object
        assert vf["document_source_url"] == s3_data.document_metadata.source_url
        assert vf["document_title"] == s3_data.document_metadata.document_title
        assert vf["family_geographies"] == s3_data.document_metadata.geographies
        assert vf["corpus_import_id"] == s3_data.document_metadata.corpus_import_id
        assert vf["corpus_type_name"] == s3_data.document_metadata.corpus_type_name
        assert vf["collection_title"] == s3_data.document_metadata.collection_title
        assert vf["collection_summary"] == s3_data.document_metadata.collection_summary

        # We expect metadata but it won't be the same shape as it is in s3
        assert isinstance(vespa_data["fields"]["metadata"], list)
        assert len(vespa_data["fields"]["metadata"]) > 0
        for metadata_item in vespa_data["fields"]["metadata"]:
            assert sorted(list(metadata_item.keys())) == ["name", "value"]


@patch.object(config, "VESPA_INSTANCE_URL", new=VESPA_TEST_ENDPOINT)
@patch.object(config, "DEVELOPMENT_MODE", new="true")
@pytest.mark.usefixtures("cleanup_test_vespa_before", "cleanup_test_vespa_after")
def test_repeated_integration(test_vespa, s3_mock, family_document_ids):
    """
    Run repeated integration tests

    First run: on the fixture dir (all docs in mock S3)
    Second run: on a single fixture shortened to test incremental runs with shorter docs
    Third run: on the same fixture, shortened but less
    Fourth Run: on the same fixture but at its original state pre shortening
    """
    NO_CHANGE_FAMILY = "CCLW.executive.10014.4470"
    CHANGE_FAMILY = "CCLW.executive.10002.4495"

    runner = CliRunner()

    # From scratch
    result = runner.invoke(
        run_as_cli,
        args=[
            s3_mock.path,
            s3_mock.inference_results_path,
            "--files-to-index",
            json.dumps(family_document_ids),
        ],
    )
    assert result.exit_code == 0, (
        f"Exception: {result.exception if result.exception else None}\n"
        f"Stdout: {result.stdout}"
        f"Trace: {traceback.print_exception(*result.exc_info)}"
    )
    search_weights_1 = get_vespa_data(
        test_vespa, SEARCH_WEIGHTS_SCHEMA, "default_weights"
    )
    no_change_family_1 = get_vespa_data(
        test_vespa, FAMILY_DOCUMENT_SCHEMA, NO_CHANGE_FAMILY
    )

    no_change_family_passages_1 = get_existing_passage_ids(test_vespa, NO_CHANGE_FAMILY)
    change_family_passages_1 = get_existing_passage_ids(test_vespa, CHANGE_FAMILY)

    # And with an incremental run that will remove some docs
    limit = 50
    s3_mock.prepare(CHANGE_FAMILY, limit)
    result = runner.invoke(
        run_as_cli,
        args=[
            s3_mock.path,
            s3_mock.inference_results_path,
            "--files-to-index",
            json.dumps([CHANGE_FAMILY]),
        ],
    )
    assert result.exit_code == 0, (
        f"Exception: {result.exception if result.exception else None}\n"
        f"Stdout: {result.stdout}"
    )

    # Allow async feed operation to elapse.
    time.sleep(5)

    # After update
    search_weights_2 = get_vespa_data(
        test_vespa, SEARCH_WEIGHTS_SCHEMA, "default_weights"
    )
    no_change_family_2 = get_vespa_data(
        test_vespa, FAMILY_DOCUMENT_SCHEMA, NO_CHANGE_FAMILY
    )

    no_change_family_passages_2 = get_existing_passage_ids(test_vespa, NO_CHANGE_FAMILY)

    change_family_passages_2 = get_existing_passage_ids(test_vespa, CHANGE_FAMILY)

    # The first embedding item is the document description
    # So the number of passages is one less than what we limited to
    expected_text_block_count = limit - 1
    assert len(change_family_passages_2) == expected_text_block_count

    # Another incremental run that will now add back some of those docs but not all
    limit = 100
    s3_mock.prepare(CHANGE_FAMILY, limit)
    result = runner.invoke(
        run_as_cli,
        args=[
            s3_mock.path,
            s3_mock.inference_results_path,
            "--files-to-index",
            json.dumps([CHANGE_FAMILY]),
        ],
    )
    assert result.exit_code == 0, (
        f"Exception: {result.exception if result.exception else None}\n"
        f"Stdout: {result.stdout}"
    )

    # Allow async feed operation to elapse.
    time.sleep(5)

    # After update
    search_weights_3 = get_vespa_data(
        test_vespa, SEARCH_WEIGHTS_SCHEMA, "default_weights"
    )
    no_change_family_3 = get_vespa_data(
        test_vespa, FAMILY_DOCUMENT_SCHEMA, NO_CHANGE_FAMILY
    )

    no_change_family_passages_3 = get_existing_passage_ids(test_vespa, NO_CHANGE_FAMILY)

    change_family_passages_3 = get_existing_passage_ids(test_vespa, CHANGE_FAMILY)

    # The first embedding item is the document description
    # So the number of passages is one less than what we limited to
    expected_text_block_count = limit - 1
    assert len(change_family_passages_3) == expected_text_block_count

    # Rerun all, adding back those docs (restore full version in mock S3)
    s3_mock.prepare(CHANGE_FAMILY, None)
    result = runner.invoke(
        run_as_cli,
        args=[
            s3_mock.path,
            s3_mock.inference_results_path,
            "--files-to-index",
            json.dumps([CHANGE_FAMILY]),
        ],
    )
    assert result.exit_code == 0, (
        f"Exception: {result.exception if result.exception else None}\n"
        f"Stdout: {result.stdout}"
    )

    # Allow async feed operation to elapse.
    time.sleep(5)

    search_weights_4 = get_vespa_data(
        test_vespa, SEARCH_WEIGHTS_SCHEMA, "default_weights"
    )
    no_change_family_4 = get_vespa_data(
        test_vespa, FAMILY_DOCUMENT_SCHEMA, NO_CHANGE_FAMILY
    )

    no_change_family_passages_4 = get_existing_passage_ids(test_vespa, NO_CHANGE_FAMILY)
    change_family_passages_4 = get_existing_passage_ids(test_vespa, CHANGE_FAMILY)

    assert search_weights_1 == search_weights_2 == search_weights_3 == search_weights_4
    assert (
        sorted(no_change_family_1)
        == sorted(no_change_family_2)
        == sorted(no_change_family_3)
        == sorted(no_change_family_4)
    )
    assert (
        sorted(no_change_family_passages_1)
        == sorted(no_change_family_passages_2)
        == sorted(no_change_family_passages_3)
        == sorted(no_change_family_passages_4)
    )

    assert sorted(change_family_passages_1) == sorted(change_family_passages_4)
    assert sorted(change_family_passages_1) != sorted(change_family_passages_2)
    assert sorted(change_family_passages_1) != sorted(change_family_passages_3)


@patch.object(config, "VESPA_INSTANCE_URL", new=VESPA_TEST_ENDPOINT)
@patch.object(config, "DEVELOPMENT_MODE", new="true")
@pytest.mark.usefixtures("cleanup_test_vespa_before", "cleanup_test_vespa_after")
def test_concept_enrichment_integration(test_vespa, s3_mock):
    """Test that we successfully index enriched concepts."""

    DOCUMENT_ID: DocumentID = DocumentID("CCLW.executive.10014.4470")
    json_path = FIXTURE_DIR / "inference_results" / f"{DOCUMENT_ID}.json"
    inference_results: list[dict[str, Any]] = json.loads(json_path.read_text())
    DOCUMENT_PASSAAGES_WITH_CONCEPTS = len(
        [
            passage_id
            for passage_id in inference_results
            if len(inference_results[passage_id]) > 0
        ]
    )

    runner = CliRunner()
    result = runner.invoke(
        run_as_cli,
        args=[
            s3_mock.path,
            s3_mock.inference_results_path,
            "--files-to-index",
            json.dumps([DOCUMENT_ID]),
        ],
    )
    assert result.exit_code == 0, (
        f"Exception: {result.exception if result.exception else None}\n"
        f"Stdout: {result.stdout}"
    )

    # Allow async feed operation to elapse
    time.sleep(5)

    response: VespaQueryResponse = test_vespa.query(
        body={
            "yql": """
                select documentid, concepts from sources document_passage
                where family_document_ref contains phrase(@family_doc_id)
            """,
            "family_doc_id": f"id:{_NAMESPACE}:family_document::{DOCUMENT_ID}",
            "presentation.summary": "search_summary",
        },
    )
    hits = response.hits
    assert (
        len(hits) > 0
    ), f"Expected passages with concepts, got 0 hits. Response: {response}"
    passage_hits_with_concepts = []
    for hit in hits:
        hit_id = hit.get("id")
        if hit_id and "family-document-passage" in hit_id:
            hit_fields = hit.get("fields")
            if hit_fields and len(hit["fields"]["concepts"]) > 0:
                passage_hits_with_concepts.append(hit_id)

    assert len(passage_hits_with_concepts) == DOCUMENT_PASSAAGES_WITH_CONCEPTS


@patch.object(config, "VESPA_INSTANCE_URL", new=VESPA_TEST_ENDPOINT)
@patch.object(config, "DEVELOPMENT_MODE", new="true")
@pytest.mark.usefixtures("cleanup_test_vespa_before", "cleanup_test_vespa_after")
def test_cleanup_on_passage_id_format_change(test_vespa, s3_mock):
    """v1 passages are replaced with v2 passages.

    When `text_block_ids` change from v1 (non-UUID, e.g. 'p_0_b_0') to v2 (UUID),
    passage IDs change format. The indexer must clean up old-format passages so they
    don't accumulate as stray documents in Vespa.
    """
    DOCUMENT_ID = "CCLW.executive.10002.4495"
    runner = CliRunner()

    # Run 1: Index with v1 `text_block_ids` (e.g. 'p_0_b_0').
    #
    # `get_passage_id` falls back to loop index → passage IDs end with integers.
    result = runner.invoke(
        run_as_cli,
        args=[
            s3_mock.path,
            s3_mock.inference_results_path,
            "--files-to-index",
            json.dumps([DOCUMENT_ID]),
        ],
    )
    assert result.exit_code == 0, (
        f"Exception: {result.exception if result.exception else None}\n"
        f"Stdout: {result.stdout}"
        f"Trace: {traceback.print_exception(*result.exc_info)}"
    )

    # Allow async feed operation to elapse
    time.sleep(5)

    # Get the passages that were just fed in
    passages_v1 = get_existing_passage_ids(
        test_vespa,
        DOCUMENT_ID,
    )
    assert len(passages_v1) > 0
    # All v1 passage IDs end with a plain integer suffix, the sequential fallback
    assert all(passage_id.rsplit(".", 1)[-1].isdigit() for passage_id in passages_v1)

    # Replace S3 fixture with UUID `text_block_ids` to simulate v2 format documents
    s3_mock.prepare_with_uuid_ids(DOCUMENT_ID)

    # Run 2: Re-index with v2 UUID text_block_ids.
    #
    # `get_passage_id` uses UUID → passage IDs end with UUIDs so old IDs become stray
    result = runner.invoke(
        run_as_cli,
        args=[
            s3_mock.path,
            s3_mock.inference_results_path,
            "--files-to-index",
            json.dumps([DOCUMENT_ID]),
        ],
    )
    assert result.exit_code == 0, (
        f"Exception: {result.exception if result.exception else None}\n"
        f"Stdout: {result.stdout}"
        f"Trace: {traceback.print_exception(*result.exc_info)}"
    )

    # Allow async feed operation to elapse
    time.sleep(5)

    passages_v2 = get_existing_passage_ids(test_vespa, DOCUMENT_ID)

    # Same number of passages — the document content is unchanged,
    # only IDs differ.
    assert len(passages_v2) == len(passages_v1)

    # Old v1 sequential passages must be gone after cleanup
    assert set(passages_v1).isdisjoint(
        set(passages_v2)
    ), "Old v1 passage IDs were not cleaned up after re-indexing with UUID IDs"

    # New passage IDs must use UUID suffixes
    def _is_uuid(s: str) -> bool:
        try:
            uuid.UUID(s)
            return True
        except ValueError:
            return False

    assert all(_is_uuid(passage_id.rsplit(".", 1)[-1]) for passage_id in passages_v2)
