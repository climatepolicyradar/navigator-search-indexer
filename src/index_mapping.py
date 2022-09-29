"""List of expected fields in OpenSearch and their types. Used to generate the index mapping and validate documents in the indexer."""

# This field is the one that documents are grouped by. Its values are eagerly loaded into memory to speed up aggregations.
DOCUMENT_ID_FIELD = "document_name_and_id"

SORTABLE_TEXT_FIELDS = {
    "document_name",
    "document_description",
}

DATE_FIELDS = {
    "document_date",
}

INTEGER_FIELDS = {"text_block_page"}

SEARCHABLE_TEXT_FIELDS = {
    "for_search_document_name",
    "for_search_document_description",
    "text",
}

EMBEDDING_FIELDS = {
    "text_embedding",
    "document_description_embedding",
}

CATEGORICAL_FIELDS = {
    "document_id",
    "md5_sum",
    "document_url",
    "document_source_url",
    "document_country_code",
    "document_country_english_shortname",
    "document_region_code",
    "document_region_english_shortname",
    "document_source_name",
    "document_type",
    "document_category",
    "document_framework_name",
    "document_hazard_name",
    "document_instrument_name",
    "document_response_name",
    "document_sector_name",
    "document_keyword",
    "document_language",
    "document_slug",
    "document_content_type",
    "text_block_coords",
}

BOOLEAN_FIELDS = {
    "translated",
}

ALL_FIELDS = (
    {DOCUMENT_ID_FIELD}
    | SORTABLE_TEXT_FIELDS
    | DATE_FIELDS
    | INTEGER_FIELDS
    | SEARCHABLE_TEXT_FIELDS
    | EMBEDDING_FIELDS
    | CATEGORICAL_FIELDS
    | BOOLEAN_FIELDS
)
