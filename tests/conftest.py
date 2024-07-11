import json
import pytest as pytest
import os
from cloudpathlib import S3Path

from pathlib import Path
from datetime import datetime

from vespa.application import Vespa
from tenacity import RetryError

from cpr_data_access.parser_models import (
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
