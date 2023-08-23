import json
import os
from typing import List, Sequence, Union

import boto3
import botocore.client
import pytest
from moto import mock_s3

from cpr_data_access.parser_models import ParserOutput, HTMLData
from cpr_data_access.pipeline_general_models import BackendDocument

from cli.test.conftest import get_text_block


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


@pytest.fixture
def s3_bucket_and_region() -> dict:
    return {
        "bucket": "test-bucket",
        "region": "eu-west-1",
    }


@pytest.fixture
def test_file_json() -> dict:
    return {
        "document_id": "test_id",
    }


@pytest.fixture
def test_prefix() -> str:
    return "test_prefix"


@pytest.fixture
def test_file_key(s3_bucket_and_region, test_prefix) -> str:
    return f"{s3_bucket_and_region['bucket']}/{test_prefix}/test_id.json"


@pytest.fixture
def pipeline_s3_objects(
    test_prefix,
    test_file_json,
):
    """
    Return a dict of s3 objects to be used in the pipeline s3 client fixture.

    This sets up a s3 bucket with the following objects:
    - A fake document with a document_id of "test_id".

    Thus, we have a document that embeddings can be generated from in s3.
    """
    return {
        f"{test_prefix}/test_id.json": bytes(
            json.dumps(test_file_json).encode("UTF-8")
        ),
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


def get_parser_output(
    html_data: Union[HTMLData, None],
    source_url: Union[str, None],
    languages: Union[Sequence[str], None],
    content_type: Union[str, None],
    translated: bool
):
    """Return a ParserOutput object with the given parameters."""
    return ParserOutput(
            document_id="test_id",
            document_metadata=BackendDocument.parse_obj(
                {
                    "publication_ts": "2013-01-01T00:00:00",
                    "name": "Dummy Name",
                    "description": "description",
                    "source_url": "http://existing.com",
                    "download_url": None,
                    "url": None,
                    "md5_sum": None,
                    "type": "EU Decision",
                    "source": "CCLW",
                    "import_id": "TESTCCLW.executive.4.4",
                    "family_import_id": "TESTCCLW.family.4.0",
                    "category": "Law",
                    "geography": "EUR",
                    "languages": [
                        "English"
                    ],
                    "metadata": {
                        "hazards": [],
                        "frameworks": [],
                        "instruments": [
                            "Capacity building|Governance"
                        ],
                        "keywords": [
                            "Adaptation"
                        ],
                        "sectors": [
                            "Economy-wide"
                        ],
                        "topics": [
                            "Adaptation"
                        ]
                    },
                    "slug": "dummy_slug"
                }
            ),
            document_name="test_name",
            document_description="test_description",
            document_source_url=source_url,   # type: ignore
            document_cdn_object="test_cdn_object",
            document_md5_sum="test_md5_sum",
            languages=languages,
            translated=translated,
            document_slug="test_slug",
            document_content_type=content_type,
            html_data=html_data,
            pdf_data=None,
        )


@pytest.fixture
def test_parser_output_array() -> List[ParserOutput]:
    """Returns an array of parser outputs with varying block types and non/valid text."""
    return [
        get_parser_output(
            html_data=HTMLData(
                has_valid_text=True,
                text_blocks=[  # type: ignore
                    get_text_block("Table"),
                    get_text_block("Text"),
                    get_text_block("Text"),
                    get_text_block("Figure"),
                    get_text_block("Text"),
                    get_text_block("Google Text Block"),
                ],
            ),
            source_url="https://www.google.com/path.html",
            languages=["test_language"],
            content_type="text/html",
            translated=True,
        ),
        get_parser_output(
            html_data=HTMLData(
                has_valid_text=False,
                text_blocks=[  # type: ignore
                    get_text_block("Table"),
                    get_text_block("Text"),
                    get_text_block("Google Text Block"),
                ],
            ),
            source_url="https://www.google.com/path.html",
            languages=["test_language"],
            content_type="text/html",
            translated=True,
        ),
    ]


@pytest.fixture
def test_parser_output_no_source_url_no_lang_no_data() -> List[ParserOutput]:
    return [
        get_parser_output(
            html_data=None,
            source_url=None,
            languages=None,
            content_type=None,
            translated=False,
        )
    ]


@pytest.fixture
def test_parser_output_source_url_no_lang_no_data() -> List[ParserOutput]:
    return [
        get_parser_output(
            html_data=None,
            source_url="https://www.example.com/files/climate-document.pdf",
            languages=None,
            content_type=None,
            translated=False
        )
    ]


@pytest.fixture
def test_parser_output_source_url_supported_lang_data() -> List[ParserOutput]:
    return [
        get_parser_output(
            html_data=HTMLData(
                has_valid_text=True,
                text_blocks=[  # type: ignore
                    get_text_block("Table"),
                    get_text_block("Google Text Block"),
                ],
            ),
            source_url="https://www.example.com/files/climate-document.pdf",
            languages=["en"],
            content_type="text/html",
            translated=False,
        )
    ]


@pytest.fixture
def test_parser_output_source_url_un_supported_lang_data() -> List[ParserOutput]:
    return [
        get_parser_output(
            html_data=HTMLData(
                has_valid_text=True,
                text_blocks=[  # type: ignore
                    get_text_block("Table"),
                    get_text_block("Google Text Block"),
                ],
            ),
            source_url="https://www.example.com/files/climate-document.pdf",
            languages=["fr"],
            content_type="text/html",
            translated=False,
        )
    ]
