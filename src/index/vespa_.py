from collections import defaultdict
import logging
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Generator,
    Mapping,
    NewType,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

from cloudpathlib import S3Path
from cpr_sdk.models.search import Concept as VespaConcept
from cpr_sdk.parser_models import ParserOutput, PDFTextBlock, VerticalFlipError
import json
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
)
from vespa.application import Vespa
from vespa.io import VespaResponse


from src import config
from src.utils import filter_on_block_type, read_npy_file


_LOGGER = logging.getLogger(__name__)
SchemaName = NewType("SchemaName", str)
DocumentID = NewType("DocumentID", str)
Coord = tuple[float, float]
TextCoords = Sequence[Coord]  # TODO: Could do better - look at data access change
TextBlockId = NewType("TextBlockId", str)
SEARCH_WEIGHTS_SCHEMA = SchemaName("search_weights")
FAMILY_DOCUMENT_SCHEMA = SchemaName("family_document")
DOCUMENT_PASSAGE_SCHEMA = SchemaName("document_passage")
_SCHEMAS_TO_PROCESS = [
    SEARCH_WEIGHTS_SCHEMA,
    FAMILY_DOCUMENT_SCHEMA,
    DOCUMENT_PASSAGE_SCHEMA,
]
# TODO: no need to parameterise now, but namespaces
# may be useful for some data separation labels later
_NAMESPACE = "doc_search"


class VespaConfigError(config.ConfigError):
    """Error for invalid vespa config."""

    pass


class VespaIndexError(config.ConfigError):
    """Error to throw during vespa indexing."""

    pass


class VespaSearchWeights(BaseModel):
    """Weights to be applied to each ranking element in searches"""

    name_weight: float
    description_weight: float
    passage_weight: float


class VespaDocumentPassage(BaseModel):
    """Document passage representation for search"""

    search_weights_ref: str
    family_document_ref: str
    text_block: str
    text_block_id: str
    text_block_type: str
    text_block_page: Optional[Annotated[int, Field(ge=0)]] = None
    text_block_coords: Optional[TextCoords] = None
    text_embedding: Annotated[list[float], 768]
    concepts: list[VespaConcept] = []


class VespaFamilyDocument(BaseModel):
    """Family-Document combined data useful for search"""

    class MetadataItem(BaseModel):
        """A single metadata object"""

        name: str
        value: str

    search_weights_ref: str
    family_name: str
    family_name_index: str
    family_description: str
    family_description_index: str
    family_description_embedding: Annotated[
        list[float], 768
    ]  # TODO: not yet enforced by pydantic
    family_import_id: str
    family_slug: str
    family_publication_ts: str
    family_publication_year: int
    family_category: str
    family_geography: str
    family_source: str
    document_import_id: str
    document_slug: str
    document_languages: Sequence[str]
    document_md5_sum: Optional[str] = None
    document_content_type: Optional[str] = None
    document_cdn_object: Optional[str] = None
    document_source_url: Optional[str] = None
    document_title: Optional[str] = None
    family_geographies: Optional[list[str]] = None
    corpus_import_id: Optional[str] = None
    corpus_type_name: Optional[str] = None
    collection_title: Optional[str] = None
    collection_summary: Optional[str] = None
    metadata: Optional[list[MetadataItem]] = None


def reshape_metadata(
    metadata: Optional[dict[str, list[str]]]
) -> Optional[list[VespaFamilyDocument.MetadataItem]]:
    if metadata is None:
        return None

    metadata_items = []
    for key, values in metadata.items():
        metadata_items.extend(
            [
                VespaFamilyDocument.MetadataItem(
                    name=key, value=str(value) if isinstance(value, int) else value
                )
                for value in values
            ]
        )
    return metadata_items


def build_vespa_family_document(
    task: ParserOutput,
    embeddings,
    search_weights_ref,
) -> VespaFamilyDocument:
    return VespaFamilyDocument(
        search_weights_ref=search_weights_ref,
        family_name=task.document_name,
        family_name_index=task.document_name,
        family_description=task.document_description,
        family_description_index=task.document_description,
        family_description_embedding=embeddings[0].tolist(),
        family_import_id=task.document_metadata.family_import_id,
        family_slug=task.document_metadata.family_slug,
        family_publication_ts=task.document_metadata.publication_ts.isoformat(),
        family_publication_year=task.document_metadata.publication_ts.year,
        family_category=task.document_metadata.category,
        family_geography=task.document_metadata.geography,
        family_source=task.document_metadata.source,
        document_import_id=task.document_id,
        document_slug=task.document_slug,
        document_languages=task.document_metadata.languages,
        document_md5_sum=task.document_md5_sum,
        document_content_type=task.document_content_type,
        document_cdn_object=task.document_cdn_object,
        document_source_url=task.document_metadata.source_url,
        document_title=task.document_metadata.document_title,
        family_geographies=task.document_metadata.geographies,
        corpus_import_id=task.document_metadata.corpus_import_id,
        corpus_type_name=task.document_metadata.corpus_type_name,
        collection_title=task.document_metadata.collection_title,
        collection_summary=task.document_metadata.collection_summary,
        metadata=reshape_metadata(task.document_metadata.metadata),
    )


def build_vespa_document_passage(
    family_document_id, search_weights_ref, text_block, embedding
) -> VespaDocumentPassage:
    fam_doc_ref = f"id:{_NAMESPACE}:family_document::{family_document_id}"
    return VespaDocumentPassage(
        family_document_ref=fam_doc_ref,
        search_weights_ref=search_weights_ref,
        text_block="\n".join(text_block.text),
        text_block_id=text_block.text_block_id,
        text_block_type=str(text_block.type),
        text_block_page=(
            text_block.page_number if isinstance(text_block, PDFTextBlock) else None
        ),
        text_block_coords=(
            text_block.coords if isinstance(text_block, PDFTextBlock) else None
        ),
        text_embedding=embedding.tolist(),
    )


def get_existing_passage_ids(
    vespa: Vespa, family_doc_id: DocumentID, offset: int = 0
) -> list[str]:
    """
    Retrieves all text blocks associated with a document

    In vespa terminology this means all document_passages for a given family_document
    """
    vespa_family_doc_id = f"id:{_NAMESPACE}:family_document::{family_doc_id}"
    max_hits = 5000

    existing_ids = []
    response = vespa.query(
        body={
            "yql": """
                select documentid from sources document_passage
                where family_document_ref contains phrase(@family_doc_id)
            """,
            "family_doc_id": vespa_family_doc_id,
            "hits": max_hits,
            "offset": offset,
            "queryProfile": "default",
        },
    )
    for hit in response.hits:
        passage_id = hit["id"].split("::")[-1]
        existing_ids.append(passage_id)

    if len(response.hits) + offset < response.number_documents_retrieved:
        # More to go
        offset = offset + max_hits
        existing_ids.extend(get_existing_passage_ids(vespa, family_doc_id, offset))

    return existing_ids


def determine_stray_ids(
    existing_doc_passage_ids: list[str], new_passage_ids: list[str]
) -> list[str]:
    return list(set(existing_doc_passage_ids) - set(new_passage_ids))


def remove_ids(vespa: Vespa, stray_ids: list[str]):
    _LOGGER.critical(f"Removing stray ids following doc changes: {stray_ids}")
    for stray_id in stray_ids:
        vespa.delete_data(
            schema=DOCUMENT_PASSAGE_SCHEMA, data_id=stray_id, namespace=_NAMESPACE
        )


def retrieve_inference_result(
    inference_results_s3_path: S3Path, document_id: str
) -> dict[TextBlockId, list[VespaConcept]] | None:
    """Retrieve an individual inference results file from S3 for a document."""

    root_document_s3_path: S3Path = inference_results_s3_path / f"{document_id}.json"
    translated_document_s3_path: S3Path = (
        inference_results_s3_path / f"{document_id}_translated_en.json"
    )

    if root_document_s3_path.exists():
        inference_result_s3_path: S3Path = root_document_s3_path
    elif translated_document_s3_path.exists():
        inference_result_s3_path: S3Path = translated_document_s3_path
    else:
        return None

    inference_result_data: dict[str, Any] = json.loads(
        inference_result_s3_path.read_text()
    )
    inference_result: dict[TextBlockId, list[VespaConcept]] = {
        TextBlockId(id): [VespaConcept.model_validate(c) for c in concepts]
        for id, concepts in inference_result_data
    }

    return inference_result


def join_concepts(
    document_passage: VespaDocumentPassage,
    inference_result: dict[
        TextBlockId,
        list[VespaConcept],
    ],
) -> VespaDocumentPassage:
    """Join Concepts from inference results on VespaDocumentPassage using the ID."""

    concepts: list[VespaConcept] = inference_result.get(
        TextBlockId(document_passage.text_block_id), []
    )

    document_passage.concepts = concepts

    return document_passage


def get_document_generator(
    vespa: Vespa,
    paths: Sequence[Union[S3Path, Path]],
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
        name_weight=2.5,
        description_weight=2.0,
        passage_weight=1.0,
    )
    yield SEARCH_WEIGHTS_SCHEMA, search_weights_id, search_weights.model_dump()

    RUNNING_AGAINST_S3: bool = isinstance(embedding_dir_as_path, S3Path)
    INFERENCE_RESULTS_S3_PATH: S3Path | None = (
        (embedding_dir_as_path.parent.parent / "inference_results/latest/")
        if RUNNING_AGAINST_S3
        else None
    )

    _LOGGER.info(
        "Filtering unwanted text block types.",
        extra={"props": {"BLOCKS_TO_FILTER": config.BLOCKS_TO_FILTER}},
    )

    search_weights_ref = f"id:{_NAMESPACE}:search_weights::{search_weights_id}"
    physical_document_count = 0
    for path in paths:
        task = ParserOutput.model_validate_json(path.read_text())

        task = filter_on_block_type(
            input=task, remove_block_types=config.BLOCKS_TO_FILTER
        )

        task_array_file_path = cast(
            Path, embedding_dir_as_path / f"{task.document_id}.npy"
        )
        embeddings = read_npy_file(task_array_file_path)

        family_document_id = DocumentID(task.document_metadata.import_id)
        family_document = build_vespa_family_document(
            task, embeddings, search_weights_ref
        )

        yield FAMILY_DOCUMENT_SCHEMA, family_document_id, family_document.model_dump()
        physical_document_count += 1
        if (physical_document_count % 50) == 0:
            _LOGGER.info(
                f"Document generator processing {physical_document_count} "
                "physical documents"
            )

        try:
            text_blocks = task.vertically_flip_text_block_coords().get_text_blocks()
        except VerticalFlipError:
            _LOGGER.exception(
                f"Error flipping text blocks for {task.document_id}, coordinates "
                "will be incorrect for displayed passages"
            )
            text_blocks = task.get_text_blocks()

        existing_doc_passage_ids = get_existing_passage_ids(vespa, family_document_id)

        new_passage_ids = []

        # Enrich the passage with concepts from S3 Inference Results if they exist so
        # as to not wipe concepts when indexing. The following is an interrim solution
        # whilst platform work towards a more complete solution to consolidate indexers.
        INFERENCE_RESULTS_MATCH: bool = False
        INFERENCE_RESULT: dict[TextBlockId, list[VespaConcept]] | None = None

        if RUNNING_AGAINST_S3 and INFERENCE_RESULTS_S3_PATH:
            try:
                INFERENCE_RESULT = retrieve_inference_result(
                    inference_results_s3_path=INFERENCE_RESULTS_S3_PATH,
                    document_id=task.document_id,
                )

                if INFERENCE_RESULT:
                    inference_result_passage_ids: list[TextBlockId] = list(
                        INFERENCE_RESULT.keys()
                    )
                    text_block_passage_ids: list[TextBlockId] = [
                        TextBlockId(text_block_id) for text_block_id in text_blocks
                    ]
                    if set(inference_result_passage_ids) == set(text_block_passage_ids):
                        INFERENCE_RESULTS_MATCH = True

            except Exception as e:
                _LOGGER.warning(
                    f"Failed to load inference results for document {task.document_id} at: "
                    f"{INFERENCE_RESULTS_S3_PATH}: {e}"
                )

        # Note that the first embedding item is the doc description
        # The rest are text blocks
        for document_passage_idx, (text_block, embedding) in enumerate(
            zip(text_blocks, embeddings[1:, :])
        ):
            document_psg_id = DocumentID(f"{task.document_id}.{document_passage_idx}")
            new_passage_ids.append(document_psg_id)
            document_passage: VespaDocumentPassage = build_vespa_document_passage(
                family_document_id, search_weights_ref, text_block, embedding
            )

            if INFERENCE_RESULT and INFERENCE_RESULTS_MATCH:
                try:
                    document_passage: VespaDocumentPassage = join_concepts(
                        document_passage, INFERENCE_RESULT
                    )
                except Exception as e:
                    _LOGGER.warning(
                        "Failed to join concepts to document passage for "
                        f"{task.document_id}: {e}"
                    )

            yield DOCUMENT_PASSAGE_SCHEMA, document_psg_id, document_passage.model_dump()
        # Cleanup stray docs
        stray_ids = determine_stray_ids(existing_doc_passage_ids, new_passage_ids)
        if stray_ids:
            remove_ids(vespa, stray_ids)

    _LOGGER.info(
        f"Document generator processed {physical_document_count} physical documents"
    )


def _check_vespa_certs():
    config_issues = []
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

    return str(key_location), str(cert_location)


def _get_vespa_instance() -> Vespa:
    """
    Creates a Vespa instance based on validated config values.

    :return Vespa: a Vespa instance to use for populating a new namespace.
    """
    if not config.VESPA_INSTANCE_URL:
        raise VespaConfigError(
            "Vespa instance URL must be configured using environment "
            "variable: 'VESPA_INSTANCE_URL'"
        )

    if config.DEVELOPMENT_MODE:
        key_location = cert_location = None
        _LOGGER.info("Running in dev mode, authentication will not be used")
    else:
        key_location, cert_location = _check_vespa_certs()

    _LOGGER.info(f"Indexing into: {config.VESPA_INSTANCE_URL}")
    return Vespa(
        url=config.VESPA_INSTANCE_URL,
        key=key_location,
        cert=cert_location,
    )


def _handle_feed_error(response: VespaResponse, id: str) -> None:
    """Callback for vespa feed"""
    if not response.is_successful():
        raise VespaIndexError(
            f"Indexing Failed on document with id: {id}, body: {response.json}"
        )


@retry(wait=wait_exponential(multiplier=10), stop=stop_after_attempt(10))
def _batch_ingest(vespa: Vespa, to_process: Mapping[SchemaName, list]):
    for schema in _SCHEMAS_TO_PROCESS:
        documents = to_process[schema]
        if documents:
            _LOGGER.info(f"Processing {schema}, with {len(documents)} documents")
            vespa.feed_iterable(
                iter=documents,
                schema=str(schema),
                namespace=_NAMESPACE,
                callback=_handle_feed_error,
                max_queue_size=(config.VESPA_DOCUMENT_BATCH_SIZE + 1),
                max_connections=config.VESPA_CONNECTIONS,
            )


def populate_vespa(
    paths: Sequence[Union[Path, S3Path]],
    embedding_dir_as_path: Union[Path, S3Path],
) -> None:
    """
    Index documents into Vespa.

    :param pdf_parser_output_dir: directory or S3 folder containing output JSON
        files from the PDF parser.
    :param embedding_dir: directory or S3 folder containing embeddings from the
        text2embeddings CLI.
    """
    vespa = _get_vespa_instance()

    document_generator = get_document_generator(
        paths=paths,
        embedding_dir_as_path=embedding_dir_as_path,
        vespa=vespa,
    )

    # Process documents into Vespa in sized groups (bulk ingest operates on documents
    # of a single schema)
    to_process: dict[SchemaName, list] = defaultdict(list)

    for schema, doc_id, fields in document_generator:
        if not fields:
            _LOGGER.critical(f"No fields for {doc_id}, of schema {schema}: {fields}")
            continue
        to_process[schema].append(
            {
                "id": doc_id,
                "fields": fields,
            }
        )

        if len(to_process[DOCUMENT_PASSAGE_SCHEMA]) >= config.VESPA_DOCUMENT_BATCH_SIZE:
            _batch_ingest(vespa, to_process)
            to_process.clear()

    _LOGGER.info("Final ingest batch")
    _batch_ingest(vespa, to_process)
