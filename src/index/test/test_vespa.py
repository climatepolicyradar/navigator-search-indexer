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
    mock_read_npy_file.return_value = read_npy_file(
        Path("src/index/test/data/CCLW.executive.10002.4495.npy")
    )

    # An array of ParserOutputs, some belonging to the same family.
    tasks = [
        get_parser_output(document_id=0, family_id=0),
        get_parser_output(document_id=1, family_id=0),
        get_parser_output(document_id=2, family_id=1),
    ]

    embedding_dir_as_path = S3Path("s3://path/to/embeddings")

    generator = get_document_generator(tasks, embedding_dir_as_path)

    vespa_family_documents = []
    vespa_document_passages = []
    for schema, id, data in generator:
        if schema == FAMILY_DOCUMENT_SCHEMA:
            vespa_family_documents.append(id)
        if schema == DOCUMENT_PASSAGE_SCHEMA:
            vespa_document_passages.append(id)

    # Check every id is unique and that there's one for each task
    assert len(set(vespa_family_documents)) == len(vespa_family_documents)
    assert len(vespa_family_documents) == len(tasks)
