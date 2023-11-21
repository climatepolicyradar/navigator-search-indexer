import json

import boto3
import botocore.client
import pytest as pytest
import os
from cloudpathlib import S3Path

from moto import mock_s3, mock_stepfunctions
from typing import Any
from pathlib import Path 

from cpr_data_access.parser_models import ParserOutput


# TODO Might be able to just mock the np.load from the path rather than all this. 

def read_local_json_file(file_path: str) -> dict:
    """Read a local json file and return the data."""
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data


def read_local_npy_file(file_path: str) -> Any:
    """Read a local npy file and return the data."""
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data


@pytest.fixture
def s3_bucket_and_region() -> dict:
    return {
        "bucket": "test-bucket",
        "region": "eu-west-1",
    }

class S3Client:
    """Helper class to connect to S3 and perform actions on buckets and documents."""

    def __init__(self, region):
        self.client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            config=botocore.client.Config(
                signature_version="s3v4",
                region_name=region,
                connect_timeout=10,
            ),
        )
     
     
# TODO: Figure out how to remove this    
@mock_stepfunctions
@pytest.fixture
def test_stepfunctions_client(s3_bucket_and_region):
    with mock_stepfunctions():
        yield boto3.client(
            "stepfunctions",
            region_name="eu-west-1",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            config=botocore.client.Config(
                signature_version="s3v4",
                region_name=s3_bucket_and_region["region"],
                connect_timeout=10,
            ),
        )


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
        str(Path(__file__).parent / os.path.join("data", "CCLW.executive.10002.4495.json"))
    )
    parser_output = ParserOutput.model_validate(parser_output_json)
    
    embeddings = read_local_npy_file(
        str(Path(__file__).parent / os.path.join("data", "CCLW.executive.10002.4495.npy"))
    )
    
    return (parser_output, embeddings)
    

@pytest.fixture
def pipeline_s3_objects(
    indexer_input_prefix: str,
    test_document_data: tuple[ParserOutput, Any]
) -> dict[str, Any]:
    """Return a dictionary of objects to load into s3."""
    parser_output, embeddings = test_document_data
    return {
        f"{indexer_input_prefix}/CCLW.executive.10002.4495.npy": embeddings,
    }


@pytest.fixture
def pipeline_s3_client(s3_bucket_and_region, pipeline_s3_objects):
    with mock_s3():
        s3_client = S3Client(s3_bucket_and_region["region"])

        s3_client.client.create_bucket(
            Bucket=s3_bucket_and_region["bucket"],
            CreateBucketConfiguration={
                "LocationConstraint": s3_bucket_and_region["region"]
            },
        )

        for key in pipeline_s3_objects:
            s3_client.client.put_object(
                Bucket=s3_bucket_and_region["bucket"],
                Key=key,
                Body=pipeline_s3_objects[key],
            )

        yield s3_client