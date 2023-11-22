import json
import pytest as pytest
import os
from cloudpathlib import S3Path

from typing import Any
from pathlib import Path
import numpy as np

from cpr_data_access.parser_models import ParserOutput


def read_local_json_file(file_path: str) -> dict:
    """Read a local json file and return the data."""
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data


def read_local_npy_file(file_path: str) -> Any:
    """Read a local npy file and return the data."""
    return np.load(file_path)


@pytest.fixture
def s3_bucket_and_region() -> dict:
    return {
        "bucket": "test-bucket",
        "region": "eu-west-1",
    }


@pytest.fixture
def indexer_input_prefix():
    return "indexer-input"


@pytest.fixture
def embeddings_dir_as_path(
    s3_bucket_and_region,
    indexer_input_prefix,
) -> S3Path:
    return S3Path(
        os.path.join("s3://", s3_bucket_and_region["bucket"], indexer_input_prefix)
    )


@pytest.fixture
def test_document_data() -> tuple[ParserOutput, Any]:
    parser_output_json = read_local_json_file(
        str(
            Path(__file__).parent
            / os.path.join("data", "CCLW.executive.10002.4495.json")
        )
    )
    parser_output = ParserOutput.model_validate(parser_output_json)

    embeddings = read_local_npy_file(
        str(
            Path(__file__).parent
            / os.path.join("data", "CCLW.executive.10002.4495.npy")
        )
    )

    return (parser_output, embeddings)
