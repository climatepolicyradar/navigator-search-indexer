import os
import logging.config

import click

from cloudpathlib import S3Path

from src.index.vespa_ import populate_vespa

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
DEFAULT_LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",  # Default is stderr
            "formatter": "json",
        },
    },
    "loggers": {},
    "root": {
        "handlers": ["console"],
        "level": LOG_LEVEL,
    },
    "formatters": {"json": {"()": "pythonjsonlogger.jsonlogger.JsonFormatter"}},
}

logging.config.dictConfig(DEFAULT_LOGGING)

os.environ["CLOUPATHLIB_FILE_CACHE_MODE"] = "close_file"


@click.command()
@click.argument("export_s3_file_path")
def run_as_cli(export_s3_file_path: str) -> None:
    """Index documents from a single pipeline_documents_for_indexing_v1 export file."""
    populate_vespa(path=S3Path(export_s3_file_path))


if __name__ == "__main__":
    run_as_cli()
