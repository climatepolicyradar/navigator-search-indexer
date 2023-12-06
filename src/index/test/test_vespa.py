import pytest
from unittest.mock import patch
from cloudpathlib import S3Path
from pathlib import Path

from src.index.vespa_ import _get_vespa_instance, VespaConfigError
from src import config
from src.index.vespa_ import (
    get_document_generator,
    FAMILY_DOCUMENT_SCHEMA,
    DOCUMENT_PASSAGE_SCHEMA,
)
from src.utils import read_npy_file
from conftest import get_parser_output


def test_get_vespa_instance() -> None:
    """Test that the get_vespa_instance function works as expected."""

    assert config.VESPA_INSTANCE_URL == ""
    expected_error_string = (
        "Vespa instance URL must be configured using environment variable: "
        "'VESPA_INSTANCE_URL'"
    )
    with pytest.raises(VespaConfigError) as context:
        _get_vespa_instance()
    assert expected_error_string in str(context.value)

    config.VESPA_INSTANCE_URL = "https://www.example.com"
    with pytest.raises(VespaConfigError) as context:
        _get_vespa_instance()
    assert expected_error_string not in str(context.value)


@patch("src.index.vespa_.read_npy_file")
def test_get_document_generator(mock_read_npy_file):
    """Assert that the vespa document generator works as expected."""
    mock_read_npy_file.return_value = read_npy_file(
        Path("src/index/test/data/CCLW.executive.10002.4495.npy")
    )

    embedding_dir_as_path = S3Path("s3://path/to/embeddings")

    # An array of ParserOutputs, some belonging to the same family.
    tasks = [
        get_parser_output(document_id=0, family_id=0),
        get_parser_output(document_id=1, family_id=0),
        get_parser_output(document_id=2, family_id=1),
    ]

    generator = get_document_generator(tasks, embedding_dir_as_path)

    vespa_family_document_ids = []
    vespa_document_passage_fam_refs = []
    for schema, id, data in generator:
        if schema == FAMILY_DOCUMENT_SCHEMA:
            vespa_family_document_ids.append(id)
        if schema == DOCUMENT_PASSAGE_SCHEMA:
            vespa_document_passage_fam_refs.append(data["family_document_ref"])

    # Check every family document id is unique and that there's one for each task
    assert len(set(vespa_family_document_ids)) == len(vespa_family_document_ids)
    assert len(vespa_family_document_ids) == len(tasks)

    # Check that every family document is referenced by one passage
    # (this is as we had one text block for each family document)
    assert len(vespa_family_document_ids) == len(vespa_document_passage_fam_refs)
    for ref in vespa_document_passage_fam_refs:
        # A document passage id CCLW.executive.0.0.0 would take the form
        # 'id:doc_search:family_document::CCLW.executive.0.0'
        ref_id_format = ref.split(":")[-1]
        assert ref_id_format in vespa_family_document_ids
