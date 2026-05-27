import logging

from cpr_sdk.parser_models import ParserOutput

from src import config

logger = logging.getLogger(__name__)


def validate_languages_decorator(func):
    """Validate that the languages requested for encoding are supported."""

    def wrapper(*args, **kwargs):
        if (
            unsupported_languages := config.TARGET_LANGUAGES
            - config.ENCODER_SUPPORTED_LANGUAGES
        ):
            logger.warning(
                f"The following languages have been requested for encoding but are not "
                f"supported by the encoder: {unsupported_languages}. Only the "
                f"following languages will be encoded: "
                f"{config.ENCODER_SUPPORTED_LANGUAGES}. "
            )
        return func(*args, **kwargs)

    return wrapper


def document_has_one_lang_that_is_supported(document: ParserOutput) -> bool:
    """Return true if the document has one language that is supported by the encoder."""
    return (
        document.languages
        and (len(document.languages) == 1)
        and (
            document.languages[0]
            in config.ENCODER_SUPPORTED_LANGUAGES.union(config.TARGET_LANGUAGES)
        )
    )


def document_has_no_source_url_languages_or_data(document: ParserOutput) -> bool:
    """Return true if the document has no source url, languages or html/pdf data."""
    return (
        not document.document_source_url
        and not document.languages
        and document.html_data is None
        and document.pdf_data is None
    )


@validate_languages_decorator
def doc_has_supported_language(document: ParserOutput) -> bool:
    """
    Identify documents that don't meet language requirements.

    Empty documents that have a source url will have a translated output produced for
    them by the pdf parser with a language that is supported by the encoder. Thus,
    we want to filter the root documents out (with no language) as we don't want to
    encode the root non-translated document as well. This is why we have the
    document_has_one_lang_that_is_supported function.
    """
    if document_has_one_lang_that_is_supported(
        document
    ) or document_has_no_source_url_languages_or_data(document):
        return True
    return False
