import functools
import json
import os
import uuid_utils as uuid
import pytest as pytest
from moto import mock_aws
import boto3
from pathlib import Path
from types import SimpleNamespace
from vespa.application import Vespa
from tenacity import RetryError

from src.index.vespa_ import _SCHEMAS_TO_PROCESS, _NAMESPACE
from src.config import VESPA_INSTANCE_URL


FIXTURE_DIR = Path(__file__).parent / "fixtures"
PIPELINE_DOCUMENTS_FOR_INDEXING_DIR = FIXTURE_DIR / "pipeline_documents_for_indexing_v1"
VESPA_TEST_ENDPOINT = os.getenv("VESPA_INSTANCE_URL", "http://localhost:8080")


@functools.lru_cache(maxsize=None)
def _pipeline_fixture_line(doc_id: str) -> str:
    """Find `doc_id`'s raw jsonl line among the pipeline_documents_for_indexing_v1 fixtures."""
    for path in PIPELINE_DOCUMENTS_FOR_INDEXING_DIR.glob("*.jsonl"):
        for line in path.read_text().splitlines():
            if line.strip() and json.loads(line)["document_id"] == doc_id:
                return line
    raise KeyError(f"No fixture row for document_id={doc_id!r}")


def get_pipeline_fixture_row(doc_id: str) -> dict:
    """Return the fixture row for `doc_id`, freshly parsed each call (safe to mutate)."""
    return json.loads(_pipeline_fixture_line(doc_id))


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


@pytest.fixture
def family_document_ids():
    """Document IDs for fixtures in tests/fixtures/pipeline_documents_for_indexing_v1."""
    return [
        "CCLW.legislative.8580.1568",  # small, has concepts, non-UUID text_block_ids
        "CCLW.document.i00003331.n0000",  # 112 passages, non-UUID text_block_ids
        "CCLW.document.i00001057.n0000",  # small, padding doc
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


def _upload_export_file(
    s3_client,
    bucket: str,
    key: str,
    family_document_ids: list[str],
    override_doc_id: str | None = None,
    limit: int | None = None,
    uuid_ids: bool = False,
) -> None:
    """Upload family_document_ids' fixture rows to moto-mocked S3 as one combined `.jsonl` file.

    Matches the real export's one-file-many-documents shape. `override_doc_id`
    applies `limit`/`uuid_ids` to just that document's row, leaving the rest unchanged.

    :param limit: truncate the override doc's vespa_document_passages to this
        many items.
    :param uuid_ids: replace the override doc's every passage text_block_id with
        a fresh UUID (v2 format).
    """
    lines = []
    for doc_id in family_document_ids:
        row = get_pipeline_fixture_row(doc_id)
        if doc_id == override_doc_id:
            if limit is not None:
                row["vespa_document_passages"] = row["vespa_document_passages"][:limit]
            if uuid_ids:
                for passage in row["vespa_document_passages"]:
                    passage["text_block_id"] = str(uuid.uuid4())
        lines.append(json.dumps(row))
    body = ("\n".join(lines) + "\n").encode()
    s3_client.put_object(Bucket=bucket, Key=key, Body=body)


@pytest.fixture
def s3_mock(s3_bucket_and_region, family_document_ids):
    """
    Mock S3 using moto. Creates bucket and populates with a combined export file.

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
        key = "pipeline_documents_for_indexing/export.jsonl"
        _upload_export_file(s3, bucket, key, family_document_ids)

        def prepare(doc_id: str, limit: int | None) -> None:
            s3_client = boto3.client("s3", region_name=s3_bucket_and_region["region"])
            _upload_export_file(
                s3_client, bucket, key, family_document_ids, doc_id, limit=limit
            )

        def prepare_with_uuid_ids(doc_id: str) -> None:
            s3_client = boto3.client("s3", region_name=s3_bucket_and_region["region"])
            _upload_export_file(
                s3_client, bucket, key, family_document_ids, doc_id, uuid_ids=True
            )

        yield SimpleNamespace(
            path=f"s3://{bucket}/{key}",
            bucket=bucket,
            region=s3_bucket_and_region["region"],
            prepare=prepare,
            prepare_with_uuid_ids=prepare_with_uuid_ids,
        )
