from typing import Optional, Sequence, Tuple, List
from enum import Enum
from datetime import date

from pydantic import BaseModel, AnyHttpUrl, Field


class ContentType(str, Enum):
    """List of document content types that can be handled by the parser."""

    HTML = "text/html"
    PDF = "application/pdf"


class DocumentMetadata(BaseModel):
    """Metadata about a document."""

    document_source_url: AnyHttpUrl
    # TODO: add other metadata fields from loader


class TextBlock(BaseModel):
    """
    Base class for a text block.

    :attribute text: list of text lines contained in the text block
    :attribute text_block_id: unique identifier for the text block
    :attribute language: language of the text block. 2-letter ISO code, optional.
    :attribute type: predicted type of the text block
    :attribute type_confidence: confidence score of the text block being of the predicted type
    """

    text: List[str]
    text_block_id: str
    language: Optional[str]
    type: str
    type_confidence: float = Field(ge=0, le=1)
    coords: Optional[List[Tuple[float, float]]] = None  # PDF only
    page_number: Optional[int] = None  # PDF only

    def to_string(self) -> str:
        """Returns the lines in a text block as a string with the lines separated by spaces."""

        return " ".join([line.strip() for line in self.text])


class PDFPageMetadata(BaseModel):
    """
    Set of metadata for a single page of a PDF document.

    :attribute dimensions: (width, height) of the page in pixels
    """

    page_number: int = Field(ge=0)
    dimensions: Tuple[float, float]


class PDFData(BaseModel):
    """
    Set of metadata unique to PDF documents.

    :attribute pages: List of pages contained in the document
    :attribute filename: Name of the PDF file, without extension
    :attribute md5sum: md5sum of PDF content
    :attribute language: list of 2-letter ISO language codes, optional. If null, the OCR processor didn't support language detection
    """

    page_metadata: Sequence[PDFPageMetadata]
    md5sum: str
    text_blocks: Sequence[TextBlock]


class HTMLData(BaseModel):
    """Set of metadata specific to HTML documents."""

    detected_title: Optional[str]
    detected_date: Optional[date]
    has_valid_text: bool
    text_blocks: Sequence[TextBlock]


class IndexerInput(BaseModel):
    """Input to the indexer. Serialised output of the document parser."""

    document_id: str
    document_metadata: DocumentMetadata
    document_name: str
    document_description: str
    document_url: AnyHttpUrl
    languages: Optional[Sequence[str]]
    translated: bool
    document_slug: str  # for better links to the frontend hopefully soon
    document_content_type: ContentType
    html_data: Optional[HTMLData] = None
    pdf_data: Optional[PDFData] = None

    def get_text_blocks(self) -> Sequence[TextBlock]:  # type: ignore
        """Returns the text blocks contained in the document."""

        if self.document_content_type == ContentType.PDF:
            return self.pdf_data.text_blocks  # type: ignore
        elif self.document_content_type == ContentType.HTML:
            return self.html_data.text_blocks  # type: ignore
