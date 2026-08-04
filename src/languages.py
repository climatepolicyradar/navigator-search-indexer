import logging
from typing import Any, Sequence

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


def document_has_one_lang_that_is_supported(document) -> bool:
    """Return true if the document has one language that is supported by the encoder.

    `document` is a VespaFamilyDocument, left unannotated to avoid a circular import.
    The Snowflake export writes language names ('English'), not ISO codes ('en'),
    so the supported sets in `config` are expressed as names to match.
    """
    return bool(
        document.document_languages
        and (len(set(document.document_languages)) == 1)
        and (
            document.document_languages[0]
            in config.ENCODER_SUPPORTED_LANGUAGES.union(config.TARGET_LANGUAGES)
        )
    )


def document_has_no_source_url_languages_or_data(
    document, passages: Sequence[Any]
) -> bool:
    """Return true if the document has no source url, languages or passages."""
    return (
        not document.document_source_url
        and not document.document_languages
        and not passages
    )


@validate_languages_decorator
def doc_has_supported_language(document, passages: Sequence[Any]) -> bool:
    """
    Identify documents that don't meet language requirements.

    Empty documents that have a source url will have a translated output produced for
    them by the pdf parser with a language that is supported by the encoder. Thus,
    we want to filter the root documents out (with no language) as we don't want to
    index the root non-translated document as well. This is why we have the
    document_has_one_lang_that_is_supported function.
    """
    if document_has_one_lang_that_is_supported(
        document
    ) or document_has_no_source_url_languages_or_data(document, passages):
        return True
    return False
