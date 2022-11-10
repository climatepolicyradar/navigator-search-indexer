from pathlib import Path
from typing import List
import logging
import tempfile
import shutil

import pandas as pd
import click

from cli.index_data import main as run_index_data
from src.base import IndexerInput, CONTENT_TYPE_HTML, CONTENT_TYPE_PDF

_LOGGER = logging.getLogger(__name__)


def get_document_ids() -> List[str]:
    """Get document IDs from the CSV file containing a sample of documents."""
    test_data = pd.read_csv(Path(__file__).parent / "test_data/documents_sample.csv")

    def get_id_from_row(row: pd.Series) -> str:
        return f"CCLW.{row['Category']}.{row['Id']}.{row['Document Id']}"

    document_ids = test_data.apply(get_id_from_row, axis=1).tolist()

    return document_ids


def trim_text_blocks(indexer_input: IndexerInput, num_blocks: int) -> IndexerInput:
    """Trim text blocks in indexer input to the specified number of blocks."""

    text_blocks = indexer_input.get_text_blocks()

    if len(text_blocks) <= num_blocks:
        return indexer_input

    trimmed_text_blocks = text_blocks[:num_blocks]

    if indexer_input.document_content_type == CONTENT_TYPE_PDF:
        indexer_input.pdf_data.text_blocks = trimmed_text_blocks  # type: ignore
    elif indexer_input.document_content_type == CONTENT_TYPE_HTML:
        indexer_input.html_data.text_blocks = trimmed_text_blocks  # type: ignore

    return indexer_input


@click.command()
@click.argument(
    "indexer_input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
def main(indexer_input_dir: Path):
    document_ids = get_document_ids()
    ids_in_input_dir = set([f.stem for f in indexer_input_dir.glob("*.json")])

    if missing_ids := set(document_ids) - ids_in_input_dir:
        _LOGGER.warning(
            f"{len(missing_ids)}/{len(document_ids)} IDs missing from input directory: {','.join(missing_ids)}"
        )

    ids_to_use = set(document_ids).intersection(ids_in_input_dir)
    paths_to_use = [indexer_input_dir / f"{id}.json" for id in ids_to_use]

    tasks = [IndexerInput.parse_raw(path.read_text()) for path in paths_to_use]

    tasks_trimmed_text = [trim_text_blocks(task, 20) for task in tasks]

    with tempfile.TemporaryDirectory() as output_dir:
        for task in tasks_trimmed_text:
            path = Path(output_dir) / f"{task.document_id}.json"
            path.write_text(task.json())
            shutil.copy(indexer_input_dir / f"{task.document_id}.npy", output_dir)

        _LOGGER.info(Path(output_dir).glob("*.json"))

        run_index_data(output_dir, s3=False, files_to_index=None)


if __name__ == "__main__":
    main()
