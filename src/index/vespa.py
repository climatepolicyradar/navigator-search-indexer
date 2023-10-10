import asyncio
import logging
from collections import defaultdict
from pathlib import Path
from typing import Generator, Mapping, NewType, Optional, Sequence, Tuple, Union

from cloudpathlib import S3Path
from cpr_data_access.parser_models import ParserOutput
from pydantic import BaseModel, ConstrainedList
from vespa.application import Vespa
from vespa.io import VespaResponse
import numpy as np

from src import config
from src.utils import filter_on_block_type


_LOGGER = logging.getLogger(__name__)
SchemaName = NewType("SchemaName", str)
DocumentID = NewType("DocumentID", str)
SEARCH_WEIGHTS_SCHEMA = SchemaName("search_weights")
FAMILY_DOCUMENT_SCHEMA = SchemaName("family_document")
DOCUMENT_PASSAGE_SCHEMA = SchemaName("document_passage")
_SCHEMAS_TO_PROCESS = [
    SEARCH_WEIGHTS_SCHEMA,
    FAMILY_DOCUMENT_SCHEMA,
    DOCUMENT_PASSAGE_SCHEMA,
]


class VespaConfigError(config.ConfigError):
    pass


class VespaIndexError(config.ConfigError):
    pass


class VespaSearchWeights(BaseModel):
    name_weight: float
    description_weight: float
    passage_weight: float


class VespaTextEmbedding(ConstrainedList):
    item_type = float
    min_items = 768
    max_items = 768


class VespaDocumentPassage(BaseModel):
    search_weights_ref: str
    family_document_ref: str
    text: str
    text_embedding: VespaTextEmbedding


class VespaFamilyDocument(BaseModel):
    search_weights_ref: str
    name: str
    description: str
    family_import_id: str
    family_slug: str
    document_import_id: str
    document_slug: str
    publication_ts: str
    category: str
    languages: Sequence[str]
    geography: str
    md5_sum: Optional[str]
    content_type: Optional[str]
    cdn_object: Optional[str]
    source_url: Optional[str]
    family_metadata: Mapping[str, Sequence[str]]
    description_embedding: VespaTextEmbedding


def get_document_generator(
    tasks: Sequence[ParserOutput],
    embedding_dir_as_path: Union[Path, S3Path],
) -> Generator[Tuple[SchemaName, DocumentID, dict], None, None]:
    """
    Get generator for documents to index.

    Documents to index are those containing text passages and their embeddings.

    :param namespace: the Vespa namespace into which these documents should be placed
    :param tasks: list of tasks from the embeddings generator
    :param embedding_dir_as_path: directory containing embeddings .npy files.
        These are named with IDs corresponding to the IDs in the tasks.
    :yield Generator[Tuple[SchemaName, DocumentID, dict], None, None]: generator of
        Vespa documents along with their schema and ID.
    """

    search_weights_id = DocumentID("default_weights")
    search_weights = VespaSearchWeights(
        name_weight = 2.5,
        description_weight = 2.0,
        passage_weight = 1.0,
    )
    yield SEARCH_WEIGHTS_SCHEMA, search_weights_id, search_weights.dict()

    _LOGGER.info(
        "Filtering unwanted text block types.",
        extra={"props": {"BLOCKS_TO_FILTER": config.BLOCKS_TO_FILTER}},
    )
    tasks = filter_on_block_type(
        inputs=tasks, remove_block_types=config.BLOCKS_TO_FILTER
    )

    physical_document_count = 0
    for task in tasks:
        embeddings = np.load(str(embedding_dir_as_path / f"{task.document_id}.npy"))

        family_document_id = DocumentID(task.document_metadata.family_import_id)
        family_document = VespaFamilyDocument(
            search_weights_ref = f"id:documents:search_weights::{search_weights_id}",
            name = task.document_name,
            description = task.document_description,
            family_import_id = task.document_metadata.family_import_id,
            family_slug = task.document_metadata.family_slug,
            publication_ts = task.document_metadata.publication_ts.isoformat(),
            document_import_id = task.document_id,
            document_slug = task.document_slug,
            category = task.document_metadata.category,
            languages = task.document_metadata.languages,
            geography = task.document_metadata.geography,
            md5_sum = task.document_md5_sum,
            content_type = task.document_content_type,
            cdn_object = task.document_cdn_object,
            source_url = task.document_metadata.source_url,
            family_metadata = task.document_metadata.metadata,
            description_embedding = embeddings[0].tolist(),
        )
        yield FAMILY_DOCUMENT_SCHEMA, family_document_id, family_document.dict()
        physical_document_count += 1
        if (physical_document_count % 50) == 0:
            _LOGGER.info(
                f"Document generator processing {physical_document_count} "
                "physical documents"
            )

        try:
            text_blocks = task.vertically_flip_text_block_coords().get_text_blocks()
        except KeyError:
            _LOGGER.exception(f"Error flipping text blocks for {task.document_id}")
            continue

        for document_passage_idx, (text_block, embedding) in enumerate(
            zip(text_blocks, embeddings[1:, :])
        ):
            fam_doc_ref = f"id:documents:family_document::{family_document_id}"
            search_weights_ref = f"id:documents:search_weights::{search_weights_id}"
            document_passage = VespaDocumentPassage(
                family_document_ref = fam_doc_ref,
                search_weights_ref = search_weights_ref,
                text = "\n".join(text_block.text),
                text_embedding = embedding.tolist(),
            )
            document_psg_id = DocumentID(f"{task.document_id}.{document_passage_idx}")
            yield DOCUMENT_PASSAGE_SCHEMA, document_psg_id, document_passage.dict()

    _LOGGER.info(
        f"Document generator processed {physical_document_count} physical documents"
    )


def _get_vespa_instance() -> Vespa:
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

    return Vespa(
        url=config.VESPA_INSTANCE_URL,
        key=str(key_location),
        cert=str(cert_location),
    )


async def _batch_ingest(vespa: Vespa, to_process: Mapping[SchemaName, list]):
    responses: list[VespaResponse] = []
    for schema in _SCHEMAS_TO_PROCESS:
        if documents := to_process[schema]:
            responses.extend(vespa.app.feed_batch(
                batch=list(documents),
                schema=str(schema),
                asynchronous=True,
                connections=50,
                batch_size=1000,
            ))

    errors = [(r.status_code, r.json) for r in responses if r.status_code >= 300]
    if errors:
        _LOGGER.error(
            "Indexing Failed",
            extra={"props": {"error_responses": errors}},
        )
        raise VespaIndexError("Indexing Failed")


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
    vespa = _get_vespa_instance()

    document_generator = get_document_generator(
        tasks=tasks,
        embedding_dir_as_path=embedding_dir_as_path,
    )

    # Process documents into Vespa in sized groups (bulk ingest operates on documents
    # of a single schema)
    to_process: dict[SchemaName, list] = defaultdict(list)

    for schema, doc_id, fields in document_generator:
        to_process[schema].append({
            "id": doc_id,
            "fields": fields,
        })

        if len(to_process[FAMILY_DOCUMENT_SCHEMA]) >= config.VESPA_DOCUMENT_BATCH_SIZE:
            asyncio.run(_batch_ingest(vespa, to_process))
            to_process.clear()

    asyncio.run(_batch_ingest(vespa, to_process))
