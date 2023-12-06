import json
import pytest as pytest
import os
from cloudpathlib import S3Path

from typing import Any
from pathlib import Path
import numpy as np
from datetime import datetime

from cpr_data_access.parser_models import (
    ParserOutput,
    BackendDocument,
    PDFData,
    PDFTextBlock,
    BlockType,
    PDFPageMetadata,
)


def read_local_json_file(file_path: str) -> dict:
    """Read a local json file and return the data."""
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data


def read_local_npy_file(file_path: str) -> Any:
    """Read a local npy file and return the data."""
    return np.load(file_path)


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
