import logging
from typing import List

from src import config
from cpr_data_access.parser_models import ParserOutput

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


def task_has_one_lang_that_is_supported(task: ParserOutput) -> bool:
    """Return true if the task has one language that is supported by the encoder."""
    return (
            task.languages
            and (len(task.languages) == 1)
            and (
                    task.languages[0]
                    in config.ENCODER_SUPPORTED_LANGUAGES.union(config.TARGET_LANGUAGES)
            )
    )


def task_has_no_source_url_languages_or_data(task: ParserOutput) -> bool:
    """Return true if the task has no source url, languages or html/pdf data."""
    return (
            not task.document_source_url
            and not task.languages
            and task.html_data is None
            and task.pdf_data is None
    )


@validate_languages_decorator
def get_docs_of_supported_language(
    tasks: List[ParserOutput],
) -> List[ParserOutput]:
    """Filter out documents that don't meet language requirements.

    Empty documents that have a source url will have a translated output produced for
    them by the pdf parser with a language that is supported by the encoder. Thus,
    we want to filter the root documents out (with no language) as we don't want to
    encode the root non-translated document as well. This is why we have the
    task_has_one_lang_that_is_supported function.
    """
    return [
        task
        for task in tasks
        if task_has_one_lang_that_is_supported(task)
        or task_has_no_source_url_languages_or_data(task)
    ]
