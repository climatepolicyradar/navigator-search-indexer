import datetime
from typing import List
import json
import os
import boto3
import botocore.client
import pytest
from moto import mock_s3

from src.base import IndexerInput, DocumentMetadata, HTMLData
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
        f'{test_prefix}/test_id.json': bytes(json.dumps(test_file_json).encode("UTF-8")),
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


@pytest.fixture
def test_indexer_input_array() -> List[IndexerInput]:
    """Test IndexerInput array with html containing various text block types."""
    return [
        IndexerInput(
            document_id="test_id",
            document_metadata=DocumentMetadata(
                publication_ts=datetime.datetime.now(),
                date="test_date",
                geography="test_geography",
                category="test_category",
                source="test_source",
                type="test_type",
                sectors=["test_sector"],
            ),
            document_name="test_name",
            document_description="test_description",
            document_source_url="https://www.google.com/path.html",
            document_cdn_object="test_cdn_object",
            document_md5_sum="test_md5_sum",
            languages=["test_language"],
            translated=True,
            document_slug="test_slug",
            document_content_type="text/html",
            html_data=HTMLData(
                has_valid_text=True,
                text_blocks=[
                    get_text_block("Table"),
                    get_text_block("Text"),
                    get_text_block("Text"),
                    get_text_block("Figure"),
                    get_text_block("Text"),
                    get_text_block("Random"),
                    get_text_block("Google Text Block"),
                ],
            ),
            pdf_data=None,
        ),
        IndexerInput(
            document_id="test_id",
            document_metadata=DocumentMetadata(
                publication_ts=datetime.datetime.now(),
                date="test_date",
                geography="test_geography",
                category="test_category",
                source="test_source",
                type="test_type",
                sectors=["test_sector"],
            ),
            document_name="test_name",
            document_description="test_description",
            document_source_url="https://www.google.com/path.html",
            document_cdn_object="test_cdn_object",
            document_md5_sum="test_md5_sum",
            languages=["test_language"],
            translated=True,
            document_slug="test_slug",
            document_content_type="text/html",
            html_data=HTMLData(
                has_valid_text=False,
                text_blocks=[
                    get_text_block("Table"),
                    get_text_block("Text"),
                    get_text_block("Google Text Block"),
                ],
            ),
            pdf_data=None,
        ),
    ]


@pytest.fixture
def test_indexer_input_no_lang() -> List[IndexerInput]:
    """Test IndexerInput array with html containing various text block types."""
    return [
        IndexerInput(
            document_id="test_id",
            document_metadata=DocumentMetadata(
                publication_ts=datetime.datetime.now(),
                date="test_date",
                geography="test_geography",
                category="test_category",
                source="test_source",
                type="test_type",
                sectors=["test_sector"],
            ),
            document_name="test_name",
            document_description="test_description",
            document_source_url="https://www.google.com/path.html",
            document_cdn_object="test_cdn_object",
            document_md5_sum="test_md5_sum",
            languages=None,
            translated=False,
            document_slug="test_slug",
            document_content_type="text/html",
            html_data=HTMLData(
                has_valid_text=True,
                text_blocks=[
                    get_text_block("Table"),
                    get_text_block("Text"),
                    get_text_block("Text"),
                    get_text_block("Figure"),
                    get_text_block("Text"),
                    get_text_block("Random"),
                    get_text_block("Google Text Block"),
                ],
            ),
            pdf_data=None,
        )
    ]


@pytest.fixture
def test_indexer_input_no_source_url() -> List[IndexerInput]:
    """Test IndexerInput array with html containing various text block types."""
    return [
        IndexerInput(
            document_id="test_id",
            document_metadata=DocumentMetadata(
                publication_ts=datetime.datetime.now(),
                date="test_date",
                geography="test_geography",
                category="test_category",
                source="test_source",
                type="test_type",
                sectors=["test_sector"],
            ),
            document_name="test_name",
            document_description="test_description",
            document_source_url=None,
            document_cdn_object=None,
            document_md5_sum="test_md5_sum",
            languages=None,
            translated=False,
            document_slug="test_slug",
            document_content_type=None,
            html_data=None,
            pdf_data=None,
        )
    ]