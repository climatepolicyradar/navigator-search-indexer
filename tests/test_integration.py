from unittest.mock import patch
import json
import traceback

from click.testing import CliRunner
from cpr_sdk.parser_models import ParserOutput
import numpy as np
import pytest
from vespa.application import Vespa

from cli.index_data import run_as_cli
from conftest import FIXTURE_DIR, VESPA_TEST_ENDPOINT
from src import config
from src.index.vespa_ import (
    SEARCH_WEIGHTS_SCHEMA,
    FAMILY_DOCUMENT_SCHEMA,
    _NAMESPACE,
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
            "--index-type",
            "vespa",
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
            "--index-type",
            "vespa",
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
            "--index-type",
            "vespa",
            "--files-to-index",
            json.dumps([CHANGE_FAMILY]),
        ],
    )
    assert result.exit_code == 0, (
        f"Exception: {result.exception if result.exception else None}\n"
        f"Stdout: {result.stdout}"
    )

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
            "--index-type",
            "vespa",
            "--files-to-index",
            json.dumps([CHANGE_FAMILY]),
        ],
    )
    assert result.exit_code == 0, (
        f"Exception: {result.exception if result.exception else None}\n"
        f"Stdout: {result.stdout}"
    )

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
            "--index-type",
            "vespa",
            "--files-to-index",
            json.dumps([CHANGE_FAMILY]),
        ],
    )
    assert result.exit_code == 0, (
        f"Exception: {result.exception if result.exception else None}\n"
        f"Stdout: {result.stdout}"
    )

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
