import pytest
from unittest.mock import patch
from datetime import datetime
from cloudpathlib import S3Path
from pathlib import Path

from cpr_data_access.parser_models import (
    ParserOutput, 
    BackendDocument, 
    PDFData, 
    PDFTextBlock, 
    BlockType,
    PDFPageMetadata
)
from src.index.vespa_ import _get_vespa_instance, VespaConfigError
from src import config
from src.index.vespa_ import get_document_generator
from src.utils import read_npy_file


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


@patch('src.index.vespa_.read_npy_file')
def test_get_document_generator(mock_read_npy_file):
    mock_read_npy_file.return_value = read_npy_file(
        Path("src/index/test/data/CCLW.executive.10002.4495.npy")
    )
    
    def get_parser_output(document_id: int, family_id: int) -> ParserOutput: 
        """
        Create a ParserOutput with specific family and document ids.
        
        Some other data is 
        """
        return ParserOutput(
            document_id=f"CCLW.executive.{document_id}.0",
            document_name="Example name",
            document_description="Example description.",
            document_slug="",
            document_content_type="application/pdf",
            pdf_data=PDFData(
                page_metadata=[
                    PDFPageMetadata(
                        page_number=1,
                        dimensions=(612.0, 792.0)
                    )   
                ],
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
                            (89.58967590332031, 303.5213928222656)
                        ],
                        page_number=1,
                    )
                ]
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
                metadata={}
            )    
        )

    # An array of ParserOutputs, some belonging to the same family. 
    tasks = [
        get_parser_output(document_id=0, family_id=0),
        get_parser_output(document_id=1, family_id=0),
        get_parser_output(document_id=2, family_id=1)
    ]

    embedding_dir_as_path = S3Path("s3://path/to/embeddings")

    generator = get_document_generator(tasks, embedding_dir_as_path)

    doc_ids = [doc[1] for doc in generator if "executive" in doc[1]]
    
    assert len(doc_ids) == len(tasks)
