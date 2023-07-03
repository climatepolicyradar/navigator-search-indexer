from typing import List

from src import config
from src.base import Text2EmbeddingsInput
from src.languages import get_docs_of_supported_language


def test_get_docs_of_supported_language(
    test_indexer_input_no_lang: List[Text2EmbeddingsInput],
    test_indexer_input_array: List[Text2EmbeddingsInput],
    test_indexer_input_no_source_url: List[Text2EmbeddingsInput],
):
    """Tests that the get_docs_of_supported_language function returns only docs of a supported language."""
    docs_of_supported_languages = get_docs_of_supported_language(
        test_indexer_input_no_lang
        + test_indexer_input_array
        + test_indexer_input_no_source_url
    )
    for doc in docs_of_supported_languages:
        assert doc.languages in config.TARGET_LANGUAGES or (
            doc.document_source_url is None
            and not doc.languages
            and doc.html_data is None
            and doc.pdf_data is None
        )


# TODO test that the warning is logged if the document language is not supported by the encoder
