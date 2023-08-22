from pathlib import Path
from typing import Optional, Sequence

import pytest

from src.base import IndexerInput, CONTENT_TYPE_HTML, CONTENT_TYPE_PDF
from src.index_mapping import ALL_FIELDS
from cli.index_data import get_core_document_generator, get_text_document_generator


@pytest.fixture()
def test_input_dir() -> Path:
    return (Path(__file__).parent / "test_data" / "index_data_input").resolve()


def test_get_core_document_generator(test_input_dir: Path):
    """Test that the document generator returns documents in the correct format."""

    tasks = [
        IndexerInput.parse_raw(path.read_text())
        for path in list(test_input_dir.glob("*.json"))
    ]

    # checking that we've picked up some tasks, otherwise the test is pointless
    # because the document generator will be empty
    assert len(tasks) > 0

    doc_generator = get_core_document_generator(tasks, test_input_dir)

    for_search_name_count = 0
    for_search_description_count = 0

    for doc in doc_generator:
        for field in [
            "document_id",
            "document_name",
            "document_name_and_slug",
            "document_description",
            "document_source_url",
            "document_cdn_object",
            "document_md5_sum",
            "translated",
            "document_slug",
            "document_content_type",
            "document_geography",
            "document_category",
            "document_source",
            "document_type",
            "document_sectors",
            "document_date",
        ]:
            assert field in doc, f"{field} not found in {doc}"

        if "for_search_document_name" in doc:
            for_search_name_count += 1
            assert all(
                [
                    field not in doc
                    for field in {
                        "for_search_document_description",
                        "document_description_embedding",
                        "text_block_id",
                        "text",
                        "text_embedding",
                    }
                ]
            )

        if "for_search_document_description" in doc:
            for_search_description_count += 1
            assert all(
                [
                    field not in doc
                    for field in {
                        "for_search_document_name",
                        "text_block_id",
                        "text",
                        "text_embedding",
                    }
                ]
            )

            assert "document_description_embedding" in doc

    assert for_search_name_count == len(tasks)
    assert for_search_description_count == len(tasks)


@pytest.mark.parametrize("translated", [True, False, None])
@pytest.mark.parametrize(
    "content_types", [[CONTENT_TYPE_PDF], [CONTENT_TYPE_HTML], None]
)
def test_get_text_document_generator(
    test_input_dir: Path,
    translated: Optional[bool],
    content_types: Optional[Sequence[str]],
):
    """Test that the document generator returns documents in the correct format."""

    tasks = [
        IndexerInput.parse_raw(path.read_text())
        for path in list(test_input_dir.glob("*.json"))
    ]

    # checking that we've picked up some tasks, otherwise the test is pointless as the document generator will be empty
    assert len(tasks) > 0

    doc_generator = get_text_document_generator(
        tasks, test_input_dir, translated=translated, content_types=content_types
    )

    for doc in doc_generator:
        for field in [
            "document_id",
            "document_name",
            "document_description",
            "document_source_url",
            "document_cdn_object",
            "document_md5_sum",
            "translated",
            "document_slug",
            "document_name_and_slug",
            "document_content_type",
            "document_sectors",
            "document_geography",
            "document_category",
            "document_source",
            "document_type",
            "document_date",
        ]:
            assert field in doc, f"{field} not found in {doc}"

        assert all(
            [
                field not in doc
                for field in {
                    "for_search_document_name",
                    "for_search_document_description",
                    "document_description_embedding",
                }
            ]
        )

        if "text_block_id" in doc:
            assert all(
                [
                    field not in doc
                    for field in {
                        "for_search_document_name",
                        "for_search_document_description",
                        "document_description_embedding",
                    }
                ]
            )

            assert "text" in doc
            assert "text_embedding" in doc
            assert "text_block_coords" in doc
            assert "text_block_page" in doc

        if translated:
            assert doc["translated"] == translated

        if content_types:
            assert doc["document_content_type"] == content_types[0]


def test_document_generator_mapping_alignment(test_input_dir: Path):
    """
    Test that the document generator only returns fields which are in the set of constants used to create the index mapping.

    This means that the document generator will not produce any fields which we're not expecting, so there will be no unpredictable type or analyzer behaviours in OpenSearch.
    """

    tasks = [
        IndexerInput.parse_raw(path.read_text())
        for path in list(test_input_dir.glob("*.json"))
    ]

    assert len(tasks) > 0  # otherwise test is pointless

    doc_generator = get_core_document_generator(tasks, test_input_dir)

    all_fields_flat = [field for fields in ALL_FIELDS.values() for field in fields]

    for doc in doc_generator:
        fields_not_in_mapping = set(doc.keys()) - set(all_fields_flat)
        assert len(fields_not_in_mapping) == 0
