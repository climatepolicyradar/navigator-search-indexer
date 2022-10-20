"""List of expected fields in OpenSearch and their types. Used to generate the index mapping and validate documents in the indexer."""
from typing import List, Dict

# Fields that should appear in every Opensearch document
COMMON_FIELDS: Dict[str, List[str]] = {
    "id": ["document_id"],
    "sortable": ["document_name", "document_description"],
    "date": [],  # ["document_date"], # TODO: uncomment when we have a date field passed from the loader
    "boolean": ["translated"],
    "categorical": [
        "document_slug",
        "document_cdn_object",
        "document_content_type",
        "document_md5_sum",
        "document_source_url",
    ],
}

# Fields that appear only in some Opensearch documents
OPTIONAL_FIELDS: Dict[str, List[str]] = {
    "integer": ["text_block_page"],
    "searchable": [
        "for_search_document_name",
        "for_search_document_description",
        "text",
    ],
    "embedding": ["text_embedding", "document_description_embedding"],
    "boolean": [],
    "categorical": ["text_block_coords", "text_block_id"],
}

# All fields - used to generate the index mapping
ALL_FIELDS = {
    x: COMMON_FIELDS.get(x, []) + OPTIONAL_FIELDS.get(x, [])
    for x in set(COMMON_FIELDS).union(OPTIONAL_FIELDS)
}
