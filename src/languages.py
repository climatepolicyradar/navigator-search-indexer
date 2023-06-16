import logging

from src import config
from src.base import Text2EmbeddingsInput

logger = logging.getLogger(__name__)

def validate_languages_decorator(func):
    """Validate that the languages requested for encoding are supported by the encoder."""

    def wrapper(*args, **kwargs):
        if (
                unsupported_languages := config.TARGET_LANGUAGES
                - config.ENCODER_SUPPORTED_LANGUAGES
        ):
            logger.warning(
                f"The following languages have been requested for encoding but are not supported by the encoder: "
                f"{unsupported_languages}. Only the following languages will be encoded: "
                f"{config.ENCODER_SUPPORTED_LANGUAGES}. "
            )
        return func(*args, **kwargs)

    return wrapper


@validate_languages_decorator
def get_docs_of_supported_language(tasks: list[Text2EmbeddingsInput]):
    """Filter out documents that don't meet language requirements.

    Persist documents with either:
     - one language where the language is in the target languages
     - no language and no content type.

    This assumes that the document name and description are in English.
    """
    return [
        task
        for task in tasks
        if (
               task.languages
               and (len(task.languages) == 1)
               and (
                   task.languages[0]
                   in config.ENCODER_SUPPORTED_LANGUAGES.union(config.TARGET_LANGUAGES)
               )
           )
           or (
                not task.languages
                and task.html_data is None
                and task.pdf_data is None
           )
    ]
