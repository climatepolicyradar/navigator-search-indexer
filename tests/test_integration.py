from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import traceback

from click.testing import CliRunner
from cpr_data_access.parser_models import ParserOutput
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


def prepare_temp_dir(doc_id: str, incremental_update_dir: str, limit):
    # Open originals
    embedding_file_name = f"{doc_id}.npy"
    family_document_file_name = f"{doc_id}.json"

    family_document_path = FIXTURE_DIR / "s3_files" / family_document_file_name
    family_document = ParserOutput.model_validate_json(family_document_path.read_text())
    embedding_path = FIXTURE_DIR / "s3_files" / embedding_file_name
    embedding = np.load(embedding_path)

    # Shorten
    family_document.pdf_data.page_metadata = family_document.pdf_data.page_metadata[
        :limit
    ]
    family_document.pdf_data.text_blocks = family_document.pdf_data.text_blocks[:limit]
    embedding = embedding[:limit]

    # Save
    dir_path = Path(incremental_update_dir)
    np.save(dir_path / embedding_file_name, embedding)
    with open(dir_path / family_document_file_name, "w") as f:
        f.write(family_document.model_dump_json())

    return doc_id


@patch.object(config, "VESPA_INSTANCE_URL", new=VESPA_TEST_ENDPOINT)
@patch.object(config, "DEVELOPMENT_MODE", new="true")
@pytest.mark.usefixtures("cleanup_test_vespa_before", "cleanup_test_vespa_after")
def test_integration(test_vespa):
    """
    Run repeated integration tests

    First run: on the fixture dir
    Second run: on a single fixture shortened to test incremental runs with shorter docs
    Third run: on the same fixture, shortened but less
    Fourth Run: on the same fixture but at its original state pre shortening
    """

    runner = CliRunner()
    s3_fixture_dir = str(FIXTURE_DIR / "s3_files")

    NO_CHANGE_FAMILY = "CCLW.executive.10014.4470"
    CHANGE_FAMILY = "CCLW.executive.10002.4495"

    # From scratch
    result = runner.invoke(
        run_as_cli,
        args=[
            s3_fixture_dir,
            "--index-type",
            "vespa",
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
    with TemporaryDirectory() as incremental_update_dir:
        limit = 50
        doc_id_to_index = prepare_temp_dir(
            CHANGE_FAMILY,
            incremental_update_dir,
            limit,
        )
        result = runner.invoke(
            run_as_cli,
            args=[
                incremental_update_dir,
                "--index-type",
                "vespa",
                "--files-to-index",
                json.dumps([doc_id_to_index]),
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

        no_change_family_passages_2 = get_existing_passage_ids(
            test_vespa, NO_CHANGE_FAMILY
        )

        change_family_passages_2 = get_existing_passage_ids(test_vespa, CHANGE_FAMILY)

        # The first embedding item is the document description
        # So the number of passages is one less than what we limited to
        expected_text_block_count = limit - 1
        assert len(change_family_passages_2) == expected_text_block_count

    # Another incremental run that will now add back some of those docs but not all
    with TemporaryDirectory() as incremental_update_dir:
        limit = 100
        doc_id_to_index = prepare_temp_dir(
            CHANGE_FAMILY,
            incremental_update_dir,
            limit,
        )
        result = runner.invoke(
            run_as_cli,
            args=[
                incremental_update_dir,
                "--index-type",
                "vespa",
                "--files-to-index",
                json.dumps([doc_id_to_index]),
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

        no_change_family_passages_3 = get_existing_passage_ids(
            test_vespa, NO_CHANGE_FAMILY
        )

        change_family_passages_3 = get_existing_passage_ids(test_vespa, CHANGE_FAMILY)

        # The first embedding item is the document description
        # So the number of passages is one less than what we limited to
        expected_text_block_count = limit - 1
        assert len(change_family_passages_3) == expected_text_block_count

    # Rerun all, adding back those docs
    result = runner.invoke(
        run_as_cli,
        args=[
            s3_fixture_dir,
            "--index-type",
            "vespa",
            "--files-to-index",
            json.dumps([doc_id_to_index]),
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
        no_change_family_1
        == no_change_family_2
        == no_change_family_3
        == no_change_family_4
    )
    assert (
        no_change_family_passages_1
        == no_change_family_passages_2
        == no_change_family_passages_3
        == no_change_family_passages_4
    )

    assert sorted(change_family_passages_1) == sorted(change_family_passages_4)
    assert sorted(change_family_passages_1) != sorted(change_family_passages_2)
    assert sorted(change_family_passages_1) != sorted(change_family_passages_3)
