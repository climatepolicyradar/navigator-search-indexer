import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Generator, NewType, Sequence, Tuple, Union

from cloudpathlib import S3Path
from cpr_data_access.parser_models import ParserOutput
from vespa.application import Vespa
import numpy as np

from src import config
from src.utils import filter_on_block_type


_LOGGER = logging.getLogger(__name__)
SchemaName = NewType("SchemaName", str)
DocumentID = NewType("DocumentID", str)
SEARCH_WEIGHTS_SCHEMA = SchemaName("search_weights")
FAMILY_DOCUMENT_SCHEMA = SchemaName("family_document")
DOCUMENT_PASSAGE_SCHEMA = SchemaName("document_passage")


class VespaConfigError(config.ConfigError):
    pass


class VespaIndex:
    """Load data into a Vespa index."""

    def __init__(
        self,
        url: str,
        key: Path,
        cert: Path,
        namespace: str,
    ):
        self._namespace = namespace
        self._app = Vespa(url=url, key=str(key), cert=str(cert))

        raise NotImplementedError("Vespa Indexing not yet implemented")


def get_document_generator(
    namespace: str,
    tasks: Sequence[ParserOutput],
    embedding_dir_as_path: Union[Path, S3Path],
) -> Generator[Tuple[SchemaName, DocumentID, dict], None, None]:
    """
    Get generator for text documents to index.

    Documents to index are those containing text passages and their embeddings.
    Optionally filter by whether text passages have been translated and/or the
    document content type.

    :param tasks: list of tasks from the document parser
    :param embedding_dir_as_path: directory containing embeddings .npy files.
        These are named with IDs corresponding to the IDs in the tasks.
    :param translated: optionally filter on whether text passages are translated
    :param content_types: optionally filter on content types
    :yield Generator[dict, None, None]: generator of Vespa documents
    """

    search_weights_id = DocumentID("default_weights")
    search_weights_fields = {
        "name_weight": 2.5,
        "description_weight": 2.0,
        "passage_weight": 1.0,
    }
    yield SEARCH_WEIGHTS_SCHEMA, search_weights_id, search_weights_fields

    _LOGGER.info(
        "Filtering unwanted text block types.",
        extra={"props": {"BLOCKS_TO_FILTER": config.BLOCKS_TO_FILTER}},
    )
    tasks = filter_on_block_type(
        inputs=tasks, remove_block_types=config.BLOCKS_TO_FILTER
    )

    for task in tasks:
        embeddings = np.load(str(embedding_dir_as_path / f"{task.document_id}.npy"))

        family_document_id = DocumentID(task.document_metadata.family_import_id)
        family_document_fields = {
            "search_weights_ref": f"id:{namespace}:search_weights::{search_weights_id}",
            "name": task.document_name,
            "family_import_id": task.document_metadata.family_import_id,
            "publication_ts": task.document_metadata.publication_ts,
            "last_updated_ts": task.document_metadata.publication_ts,  # TODO: more data
            "document_import_id": task.document_id,
            "document_metadata": task.document_metadata,
            "languages": task.document_metadata.languages,
            "geography": task.document_metadata.geography,
            "md5_sum": task.document_md5_sum,
            "cdn_object": task.document_cdn_object,
            "source_url": task.document_metadata.source_url,
            "content_type": task.document_content_type,
            "description": task.document_description,
            "description_embedding": {"values": embeddings[0, :].tolist()},
        }
        yield FAMILY_DOCUMENT_SCHEMA, family_document_id, family_document_fields

        text_blocks = task.vertically_flip_text_block_coords().get_text_blocks()
        for document_passage_idx, (text_block, embedding) in enumerate(
            zip(text_blocks, embeddings[1:, :])
        ):
            fam_doc_ref = f"id:{namespace}:family_document::{family_document_id}"
            search_weights_ref = f"id:{namespace}:search_weights::{search_weights_id}"
            document_passage_fields = {
                "family_document_ref": fam_doc_ref,
                "search_weights_ref": search_weights_ref,
                "text": text_block.text,
                "text_embedding": embedding,
            }
            document_psg_id = DocumentID(f"{task.document_id}.{document_passage_idx}")
            yield DOCUMENT_PASSAGE_SCHEMA, document_psg_id, document_passage_fields


def _get_vespa_instance(namespace: str) -> VespaIndex:
    """
    Creates a Vespa instance based on validated config values.

    :return Vespa: a Vespa instance to use for populating a new namespace.
    """
    config_issues = []
    if not config.VESPA_INSTANCE_URL:
        config_issues.append(
            "Vespa instance URL must be configured using environment "
            "variable: 'VESPA_INSTANCE_URL'"
        )

    if not config.VESPA_KEY_LOCATION:
        config_issues.append(
            "Vespa key location must be configured using environment "
            "variable: 'VESPA_KEY_LOCATION'"
        )
    key_location = Path(config.VESPA_KEY_LOCATION)
    if not (key_location.exists() or key_location.is_file()):
        config_issues.append(
            "Configured key location does not exist or is not a file: "
            f"variable: '{config.VESPA_KEY_LOCATION}'"
        )

    if not config.VESPA_CERT_LOCATION:
        config_issues.append(
            "Vespa instance URL must be configured using environment "
            "variable: 'VESPA_CERT_LOCATION'"
        )
    cert_location = Path(config.VESPA_CERT_LOCATION)
    if not (cert_location.exists() or cert_location.is_file()):
        config_issues.append(
            "Configured cert location does not exist or is not a file: "
            f"variable: '{config.VESPA_CERT_LOCATION}'"
        )

    if config_issues:
        raise VespaConfigError(f"Vespa configuration issues found: {config_issues}")

    return VespaIndex(
        url=config.VESPA_INSTANCE_URL,
        key=key_location,
        cert=cert_location,
        namespace=namespace,
    )


def _batch_ingest(vespa: VespaIndex, to_process: dict):



def populate_vespa(
    tasks: Sequence[ParserOutput],
    embedding_dir_as_path: Union[Path, S3Path],
) -> None:
    """
    Index documents into Opensearch.

    :param pdf_parser_output_dir: directory or S3 folder containing output JSON
        files from the PDF parser.
    :param embedding_dir: directory or S3 folder containing embeddings from the
        text2embeddings CLI.
    """
    formatted_date = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    namespace = f"{config.VESPA_NAMESPACE_PREFIX}_{formatted_date}"
    vespa = _get_vespa_instance(namespace)

    document_generator = get_document_generator(
        namespace=namespace,
        tasks=tasks,
        embedding_dir_as_path=embedding_dir_as_path,
    )

    # Process documents into Vespa in sized groups (bulk ingest operates on documents
    # of a single schema)
    to_process = defaultdict(list)

    for schema, doc_id, fields in document_generator:
        to_process[schema].append({
            "id": doc_id,
            "fields": fields,
        })

        if len(to_process[FAMILY_DOCUMENT_SCHEMA]) >= config.VESPA_DOCUMENT_BATCH_SIZE:
            _batch_ingest(vespa, to_process)
            to_process.clear()

    _batch_ingest(vespa, to_process)
