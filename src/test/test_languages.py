from typing import List

from cpr_data_access.parser_models import ParserOutput
from src.languages import get_docs_of_supported_language

# TODO test that the warning is logged if the document language is not supported by
#  the encoder


def test_get_docs_of_supported_language(
    test_parser_output_no_source_url_no_lang_no_data: List[ParserOutput],
    test_parser_output_source_url_no_lang_no_data: List[ParserOutput],
    test_parser_output_source_url_supported_lang_data: List[ParserOutput],
    test_parser_output_source_url_un_supported_lang_data: List[ParserOutput],
):
    """Tests that the function returns only docs of a supported language."""
    assert (
        get_docs_of_supported_language(test_parser_output_no_source_url_no_lang_no_data)
        == test_parser_output_no_source_url_no_lang_no_data
    )

    assert (
        get_docs_of_supported_language(test_parser_output_source_url_no_lang_no_data)
        == []
    )

    assert (
        get_docs_of_supported_language(
            test_parser_output_source_url_supported_lang_data
        )
        == test_parser_output_source_url_supported_lang_data
    )

    assert (
        get_docs_of_supported_language(
            test_parser_output_source_url_un_supported_lang_data
        )
        == []
    )
