from cpr_sdk.parser_models import ParserOutput
from src.languages import doc_has_supported_language

# TODO test that the warning is logged if the document language is not supported by
#  the encoder


def test_doc_has_supported_language(
    test_parser_output_no_source_url_no_lang_no_data: ParserOutput,
    test_parser_output_source_url_no_lang_no_data: ParserOutput,
    test_parser_output_source_url_supported_lang_data: ParserOutput,
    test_parser_output_source_url_un_supported_lang_data: ParserOutput,
):
    """Tests that the function returns only docs of a supported language."""
    assert (
        doc_has_supported_language(test_parser_output_no_source_url_no_lang_no_data)
        is True
    )

    assert (
        doc_has_supported_language(test_parser_output_source_url_no_lang_no_data)
        is False
    )

    assert (
        doc_has_supported_language(test_parser_output_source_url_supported_lang_data)
        is True
    )

    assert (
        doc_has_supported_language(test_parser_output_source_url_un_supported_lang_data)
        is False
    )
