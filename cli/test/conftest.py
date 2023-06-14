import json
import os
from pathlib import Path

import boto3
import botocore.client
import pytest
from moto import mock_s3


@pytest.fixture()
def test_input_dir() -> Path:
    return (Path(__file__).parent / "test_data" / "text2embeddings_input").resolve()


@pytest.fixture()
def test_input_dir_bad_data() -> Path:
    return (Path(__file__).parent / "test_data" / "text2embeddings_input_bad").resolve()


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
def input_prefix() -> str:
    return "embeddings_input_test"


@pytest.fixture
def output_prefix() -> str:
    return "embeddings_output_test"


@pytest.fixture
def test_file_name() -> str:
    return "CCLWTEST.executive.1.1.json"


@pytest.fixture
def test_file_key(s3_bucket_and_region, input_prefix, test_file_name) -> str:
    return "embeddings_input_test/test.txt"


@pytest.fixture
def test_input_dir_s3(s3_bucket_and_region, input_prefix) -> str:
    return f"s3://{s3_bucket_and_region}/{input_prefix}/"


@pytest.fixture
def test_output_dir_s3(s3_bucket_and_region, output_prefix) -> str:
    return f"s3://{s3_bucket_and_region}/{output_prefix}/"


@pytest.fixture
def test_file_json() -> dict:
    return {
        "document_id": "test_html",
        "document_name": "test_html",
        "document_description": "test_html_description",
        "document_source_url": "https://www.industry.gov.au/funding-and-incentives/emissions-reduction-fund",
        "document_cdn_object": None,
        "document_md5_sum": None,
        "languages": [
            "en"
        ],
        "document_metadata": {
            "publication_ts": "2022-10-25 12:43:00.869045",
            "geography": "test_geo",
            "category": "test_category",
            "source": "test_source",
            "type": "test_type",
            "sectors": ["sector1", "sector2"]
        },
        "translated": False,
        "document_slug": "YYY",
        "document_content_type": "text/html",
        "html_data": {
            "detected_title": "Machinery of Government (MoG) changes to our department from 1 July 2022",
            "detected_date": "2020-10-22",
            "has_valid_text": False,
            "text_blocks": [
                {
                    "text": [
                        "From 1 July 2022, the Department of Industry, Science, Energy and Resources (DISER) "
                        "becomes the Department of Industry, Science and Resources (DISR). This follows "
                        "Administrative Arrangements Orders issued on 1 June 2022. "
                    ],
                    "text_block_id": "b0",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0
                },
                {
                    "text": [
                        "Department functions"
                    ],
                    "text_block_id": "b1",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0
                },
                {
                    "text": [
                        "The climate change and energy functions that previously sat with our department have "
                        "transferred to the new Department of Climate Change, Energy, the Environment and Water "
                        "(DCCEW). "
                    ],
                    "text_block_id": "b2",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0
                },
                {
                    "text": [
                        "The Department of Industry, Science and Resources retains the industry, science and "
                        "resources functions. It also takes on several functions that previously sat with the "
                        "department of Prime Minister and Cabinet. "
                    ],
                    "text_block_id": "b3",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0
                },
                {
                    "text": [
                        "Our department’s organisational chart outlines the new structure and key staff changes."
                    ],
                    "text_block_id": "b4",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0
                },
                {
                    "text": [
                        "Visit the new Department of Climate Change, Energy, the Environment and Water "
                        "dcceew.gov.au website to learn more about its remit and structure. "
                    ],
                    "text_block_id": "b5",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0
                },
                {
                    "text": [
                        "Ministers"
                    ],
                    "text_block_id": "b6",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0
                },
                {
                    "text": [
                        "The Minister for Climate Change and Energy and Assistant Minister for Climate Change "
                        "and Energy will be responsible for these portfolios in the Department of Climate "
                        "Change, Energy, the Environment and Water. "
                    ],
                    "text_block_id": "b7",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0
                },
                {
                    "text": [
                        "Visit the new minister.dcceew.gov.au website to find their media releases, speeches and "
                        "transcripts. "
                    ],
                    "text_block_id": "b8",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0
                },
                {
                    "text": [
                        "The Minister for Industry and Science, Minister for Resources, Assistant Minister for "
                        "Manufacturing and Assistant Minister for Trade will stay with our department. "
                    ],
                    "text_block_id": "b9",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0
                },
                {
                    "text": [
                        "Visit the current minister.industry.gov.au website to find their media releases, "
                        "speeches and transcripts. "
                    ],
                    "text_block_id": "b10",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0
                },
                {
                    "text": [
                        "Portfolio bodies and offices"
                    ],
                    "text_block_id": "b11",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0
                },
                {
                    "text": [
                        "A number of portfolio bodies and offices also transfer as a result of these MoG changes."
                    ],
                    "text_block_id": "b12",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0
                },
                {
                    "text": [
                        "Agencies or entities moving from the Department of Industry, Science, Energy and "
                        "Resources to the Department of Climate Change, Energy, the Environment and Water are: "
                    ],
                    "text_block_id": "b13",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0
                },
                {
                    "text": [
                        "Offices or taskforces moving from the Department of the Prime Minister and Cabinet to "
                        "the Department of Industry, Science and Resources are: "
                    ],
                    "text_block_id": "b14",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0
                },
                {
                    "text": [
                        "Accessing website content"
                    ],
                    "text_block_id": "b15",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0
                },
                {
                    "text": [
                        "Emissions reduction and energy content previously on industry.gov.au has transferred to "
                        "dcceew.gov.au. Consultations currently on consult.industry.gov.au will move later. "
                    ],
                    "text_block_id": "b16",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0
                },
                {
                    "text": [
                        "You can find all related content via the links below."
                    ],
                    "text_block_id": "b17",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0
                },
                {
                    "text": [
                        "Climate change content"
                    ],
                    "text_block_id": "b18",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0
                },
                {
                    "text": [
                        "Policies and initiatives:"
                    ],
                    "text_block_id": "b19",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0
                },
                {
                    "text": [
                        "Regulations and standards:"
                    ],
                    "text_block_id": "b20",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0
                },
                {
                    "text": [
                        "Offices and committees:"
                    ],
                    "text_block_id": "b21",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0
                },
                {
                    "text": [
                        "Publications:"
                    ],
                    "text_block_id": "b22",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0
                },
                {
                    "text": [
                        "News:"
                    ],
                    "text_block_id": "b23",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0
                },
                {
                    "text": [
                        "Consultations:"
                    ],
                    "text_block_id": "b24",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0
                },
                {
                    "text": [
                        "Energy content"
                    ],
                    "text_block_id": "b25",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0
                },
                {
                    "text": [
                        "Policies and initiatives:"
                    ],
                    "text_block_id": "b26",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0
                },
                {
                    "text": [
                        "Regulations and standards:"
                    ],
                    "text_block_id": "b27",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0
                },
                {
                    "text": [
                        "Committees"
                    ],
                    "text_block_id": "b28",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0
                },
                {
                    "text": [
                        "Publications:"
                    ],
                    "text_block_id": "b29",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0
                },
                {
                    "text": [
                        "News:"
                    ],
                    "text_block_id": "b30",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0
                },
                {
                    "text": [
                        "Consultations:"
                    ],
                    "text_block_id": "b31",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0
                },
                {
                    "text": [
                        "Finding social media channels"
                    ],
                    "text_block_id": "b32",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0
                },
                {
                    "text": [
                        "These social media channels are now part of the Department of Climate Change, Energy, "
                        "the Environment and Water: "
                    ],
                    "text_block_id": "b33",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0
                },
                {
                    "text": [
                        "During the transition, you can still find climate change and energy videos on our "
                        "department’s YouTube channel. "
                    ],
                    "text_block_id": "b34",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0
                },
                {
                    "text": [
                        "Signing up to newsletters"
                    ],
                    "text_block_id": "b35",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0
                },
                {
                    "text": [
                        "These newsletters are now part of the Department of Climate Change, Energy, "
                        "the Environment and Water: "
                    ],
                    "text_block_id": "b36",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0
                },
                {
                    "text": [
                        "Over time we will transfer any newsletters, signup forms and subscriber lists we "
                        "currently host to the new department. "
                    ],
                    "text_block_id": "b37",
                    "language": "en",
                    "type": "Text",
                    "type_confidence": 1.0
                }
            ]
        },
        "pdf_data": None
    }


@pytest.fixture
def pipeline_s3_objects(
    test_file_key,
    test_file_json,
):
    """
    Return a dict of s3 objects to be used in the pipeline s3 client fixture.

    This sets up a s3 bucket with the following objects:
    - Test parser output document

    Thus, we have a document that embeddings can be generated from in s3.
    """
    return {
        test_file_key: bytes(json.dumps(test_file_json).encode("UTF-8")),
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
