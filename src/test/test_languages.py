from typing import List

from src.base import Text2EmbeddingsInput
from src.languages import get_docs_of_supported_language

# TODO test that the warning is logged if the document language is not supported by the encoder


def test_get_docs_of_supported_language(
    test_indexer_input_no_source_url_no_lang_no_data: List[Text2EmbeddingsInput],
    test_indexer_input_source_url_no_lang_no_data: List[Text2EmbeddingsInput],
    test_indexer_input_source_url_supported_lang_data: List[Text2EmbeddingsInput],
    test_indexer_input_source_url_un_supported_lang_data: List[Text2EmbeddingsInput],
):
    """Tests that the get_docs_of_supported_language function returns only docs of a supported language."""
    assert get_docs_of_supported_language(
        test_indexer_input_no_source_url_no_lang_no_data
    ) == test_indexer_input_no_source_url_no_lang_no_data

    assert get_docs_of_supported_language(
        test_indexer_input_source_url_no_lang_no_data
    ) == []

    assert get_docs_of_supported_language(
        test_indexer_input_source_url_supported_lang_data
    ) == test_indexer_input_source_url_supported_lang_data

    assert get_docs_of_supported_language(
        test_indexer_input_source_url_un_supported_lang_data
    ) == []
