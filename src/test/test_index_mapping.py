from src.base import IndexerInput, DocumentMetadata
from src.index_mapping import COMMON_FIELDS


def test_all_common_fields_present_in_pydantic_model():
    """Check that the task model used to validate the input and the fields used to create the index mapping are aligned."""

    metadata_fields_in_indexer_input = set(
        IndexerInput.schema()["required"] + DocumentMetadata.schema()["required"]
    )
    common_fields_flat = set(
        [field for fields in COMMON_FIELDS.values() for field in fields]
    )

    missing_fields_in_schema = common_fields_flat - metadata_fields_in_indexer_input

    assert len(missing_fields_in_schema) == 0
