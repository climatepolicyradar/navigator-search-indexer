import json
import os
import uuid_utils as uuid
import pytest as pytest
from moto import mock_aws
import boto3
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace

from vespa.application import Vespa
from tenacity import RetryError

from cpr_sdk.parser_models import (
    ParserOutput,
    BackendDocument,
    PDFData,
    PDFTextBlock,
    BlockType,
    PDFPageMetadata,
)
from src.index.vespa_ import _SCHEMAS_TO_PROCESS, _NAMESPACE
from src.config import VESPA_INSTANCE_URL


FIXTURE_DIR = Path(__file__).parent / "fixtures"
INFERENCE_RESULTS_DIR = FIXTURE_DIR / "inference_results"
VESPA_TEST_ENDPOINT = os.getenv("VESPA_INSTANCE_URL", "http://localhost:8080")


def pytest_configure(config):
    cloud_url_substring = "vespa-app.cloud"
    if (
        cloud_url_substring in VESPA_INSTANCE_URL
        or cloud_url_substring in VESPA_TEST_ENDPOINT
    ):
        pytest.exit(
            "Vespa instance url looks like a cloud url: "
            f"{VESPA_INSTANCE_URL} | {VESPA_TEST_ENDPOINT} "
            "Has something been misconfigured?"
        )


def get_parser_output(document_id: int, family_id: int) -> ParserOutput:
    """Create a ParserOutput with specific family and document ids."""
    return ParserOutput(
        document_id=f"CCLW.executive.{document_id}.0",
        document_name="Example name",
        document_description="Example description.",
        document_slug="",
        document_content_type="application/pdf",
        pdf_data=PDFData(
            page_metadata=[PDFPageMetadata(page_number=1, dimensions=(612.0, 792.0))],
            md5sum="123",
            text_blocks=[
                PDFTextBlock(
                    text=[f"Example text for CCLW.executive.{document_id}.0"],
                    text_block_id="p_1_b_0",
                    type=BlockType.TEXT,
                    type_confidence=1.0,
                    coords=[
                        (89.58967590332031, 243.0702667236328),
                        (519.2817077636719, 243.0702667236328),
                        (519.2817077636719, 303.5213928222656),
                        (89.58967590332031, 303.5213928222656),
                    ],
                    page_number=1,
                )
            ],
        ),
        document_metadata=BackendDocument(
            name="Example name",
            description="Example description.",
            import_id=f"CCLW.executive.{document_id}.0",
            slug="",
            family_import_id=f"CCLW.family.{family_id}.0",
            family_slug="",
            publication_ts=datetime.now(),
            type="",
            source="",
            category="",
            geography="",
            languages=[],
            metadata={},
        ),
    )


@pytest.fixture
def family_document_ids():
    """Document IDs for integration test fixtures in s3_files."""
    return [
        "CCLW.executive.10014.4470",
        "CCLW.executive.10002.4495",
        "CCLW.document.i00000004.n0000",
    ]


@pytest.fixture
def s3_bucket_and_region() -> dict:
    return {
        "bucket": "test-bucket",
        "region": "eu-west-1",
    }


@pytest.fixture
def test_vespa():
    yield Vespa(url=VESPA_TEST_ENDPOINT)


@pytest.fixture
def preload_fixtures(test_vespa):
    for schema in _SCHEMAS_TO_PROCESS:
        fixture_path = FIXTURE_DIR / "vespa_documents" / f"{schema}.json"
        with open(fixture_path) as docs_file:
            batch = json.loads(docs_file.read())
        try:
            test_vespa.feed_iterable(iter=batch, schema=schema, namespace=_NAMESPACE)
        except RetryError as e:
            pytest.exit(reason=e.last_attempt.exception())


def cleanup_test_vespa(test_vespa):
    for schema in _SCHEMAS_TO_PROCESS:
        test_vespa.delete_all_docs(
            content_cluster_name="family-document-passage",
            schema=schema,
            namespace=_NAMESPACE,
        )


@pytest.fixture
def cleanup_test_vespa_after(test_vespa):
    yield
    cleanup_test_vespa(test_vespa)


@pytest.fixture
def cleanup_test_vespa_before(test_vespa):
    cleanup_test_vespa(test_vespa)
    yield


def _upload_s3_doc(
    s3_client,
    bucket: str,
    prefix: str,
    doc_id: str,
    limit: int | None = None,
    uuid_ids: bool = False,
) -> None:
    """Upload a fixture doc to moto-mocked S3.

    :param limit: truncate text blocks to this many items.
    :param uuid_ids: replace every text_block_id with a fresh UUID (v2 format).
    """
    json_path = FIXTURE_DIR / "s3_files" / f"{doc_id}.json"
    family_document = ParserOutput.model_validate_json(json_path.read_text())

    if limit is not None:
        family_document.pdf_data.page_metadata = family_document.pdf_data.page_metadata[
            :limit
        ]
        family_document.pdf_data.text_blocks = family_document.pdf_data.text_blocks[
            :limit
        ]

    if uuid_ids:
        for block in family_document.pdf_data.text_blocks:
            block.text_block_id = str(uuid.uuid4())

    s3_client.put_object(
        Bucket=bucket,
        Key=f"{prefix}/{doc_id}.json",
        Body=family_document.model_dump_json().encode(),
    )


@pytest.fixture
def s3_mock(s3_bucket_and_region, family_document_ids):
    """
    Mock S3 using moto. Creates bucket and populates with fixture docs.

    Yields object with .path (S3 URI for CLI) and .prepare(doc_id, limit) for overwrites.
    """
    with mock_aws():
        s3 = boto3.client(
            "s3",
            region_name=s3_bucket_and_region["region"],
        )
        bucket = s3_bucket_and_region["bucket"]
        s3.create_bucket(
            Bucket=bucket,
            CreateBucketConfiguration={
                "LocationConstraint": s3_bucket_and_region["region"],
            },
        )
        prefix = "indexer-input"
        for doc_id in family_document_ids:
            json_path = FIXTURE_DIR / "s3_files" / f"{doc_id}.json"
            s3.put_object(
                Bucket=bucket,
                Key=f"{prefix}/{doc_id}.json",
                Body=json_path.read_bytes(),
            )

        inference_results_prefix = "inference-results"
        inference_result_path = INFERENCE_RESULTS_DIR / f"{family_document_ids[0]}.json"
        s3.put_object(
            Bucket=bucket,
            Key=f"{inference_results_prefix}/{family_document_ids[0]}.json",
            Body=inference_result_path.read_bytes(),
        )

        def prepare(doc_id: str, limit: int | None) -> None:
            s3_client = boto3.client("s3", region_name=s3_bucket_and_region["region"])
            _upload_s3_doc(s3_client, bucket, prefix, doc_id, limit=limit)

        def prepare_with_uuid_ids(doc_id: str) -> None:
            s3_client = boto3.client("s3", region_name=s3_bucket_and_region["region"])
            _upload_s3_doc(s3_client, bucket, prefix, doc_id, uuid_ids=True)

        inference_results_path = f"s3://{bucket}/{inference_results_prefix}"
        yield SimpleNamespace(
            path=f"s3://{bucket}/{prefix}",
            inference_results_path=inference_results_path,
            bucket=bucket,
            region=s3_bucket_and_region["region"],
            prepare=prepare,
            prepare_with_uuid_ids=prepare_with_uuid_ids,
        )
