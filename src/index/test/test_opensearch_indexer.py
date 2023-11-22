from typing import Generator
from cloudpathlib import S3Path
from unittest.mock import patch

from src.index.opensearch import get_text_document_generator
from cpr_data_access.pipeline_general_models import CONTENT_TYPE_PDF


@patch("src.index.opensearch.np.load")
def test_get_text_document_generator(
    mock_np_load,
    test_document_data,
    embeddings_dir_as_path: S3Path,
) -> None:
    """
    Test that the generator successfully represents json files.

    Particularly page numbers.
    """
    # TODO Test that we successfully filter for translated
    # TODO test that we correctly filter for content-type
    # TODO Test that we successfully remove the correct block types
    # TODO Test that we successfully preserve page numbers
    #   (or +1 if that's the desired functionality)
    # TODO Test the keys of the returned document dictionary

    parser_output, embeddings = test_document_data
    assert parser_output.document_content_type == CONTENT_TYPE_PDF

    mock_np_load.return_value = embeddings

    text_document_generator = get_text_document_generator(
        tasks=[parser_output],
        embedding_dir_as_path=embeddings_dir_as_path,
        translated=False,
        content_types=[CONTENT_TYPE_PDF],
    )

    assert isinstance(text_document_generator, Generator)

    document = next(text_document_generator, None)

    assert document is not None
    assert isinstance(document, dict)

    parser_output_tb_pages = {
        block.page_number for block in parser_output.pdf_data.text_blocks
    }
    parser_output_md_pages = {
        page.page_number for page in parser_output.pdf_data.page_metadata
    }

    # All text block pages should exist in the page metadata object. Not all metadata
    # object pages should have a text block page as we may not have retrieved text for
    # every page.
    for text_block_page in parser_output_tb_pages:
        assert text_block_page in parser_output_md_pages

    document_pages = {doc["text_block_page"] for doc in text_document_generator}
    assert document_pages == parser_output_tb_pages

    # We expect the generator to only yield one item
    document = next(text_document_generator, None)
    assert document is None


# TODO test the other generators
