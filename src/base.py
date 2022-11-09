from typing import Optional, Sequence, Tuple, List
import datetime

from pydantic import BaseModel, AnyHttpUrl, Field, root_validator

CONTENT_TYPE_HTML = "text/html"
CONTENT_TYPE_PDF = "application/pdf"


class DocumentMetadata(BaseModel):
    """Metadata about a document."""

    publication_ts: Optional[datetime.datetime]
    date: Optional[str] = None  # Set on import by a validator
    geography: str
    category: str
    source: str
    type: str

    @root_validator
    def convert_publication_ts_to_date(cls, values):
        """Convert publication_ts to a datetime string. This is necessary as OpenSearch expects a date object."""

        values["date"] = values["publication_ts"].strftime("%d/%m/%Y")

        return values


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
    detected_date: Optional[datetime.date]
    has_valid_text: bool
    text_blocks: Sequence[TextBlock]


class IndexerInput(BaseModel):
    """Input to the indexer. Serialised output of the document parser."""

    document_id: str
    document_metadata: DocumentMetadata
    document_name: str
    document_description: str
    document_source_url: Optional[AnyHttpUrl]
    document_cdn_object: Optional[str]
    document_md5_sum: Optional[str]
    languages: Optional[Sequence[str]]
    translated: bool
    document_slug: str  # for better links to the frontend hopefully soon
    document_content_type: Optional[str]
    html_data: Optional[HTMLData] = None
    pdf_data: Optional[PDFData] = None

    def vertically_flip_text_block_coords(self) -> "IndexerInput":
        """Flips the coordinates of all PDF text blocks vertically. Acts in-place on the coordinates in the IndexerInput object."""

        # TODO: move this to the document parser

        if self.pdf_data is None:
            return self

        page_height_map = {
            page.page_number: page.dimensions[1] for page in self.pdf_data.page_metadata
        }

        for text_block in self.pdf_data.text_blocks:
            if text_block.coords is not None and text_block.page_number is not None:
                text_block.coords = [
                    (x, page_height_map[text_block.page_number] - y)
                    for x, y in text_block.coords
                ]

                # flip top and bottom so y values are still increasing as you go through the coordinates list
                text_block.coords = [
                    text_block.coords[2],
                    text_block.coords[3],
                    text_block.coords[0],
                    text_block.coords[1],
                ]

        return self

    def get_text_blocks(self) -> Sequence[TextBlock]:  # type: ignore
        """Returns the text blocks contained in the document."""

        if self.document_content_type is None:
            return []
        elif self.document_content_type == CONTENT_TYPE_PDF:
            return self.pdf_data.text_blocks  # type: ignore
        elif self.document_content_type == CONTENT_TYPE_HTML:
            if self.html_data.has_valid_text:  # type: ignore
                return self.html_data.text_blocks  # type: ignore
            else:
                return []

    @root_validator
    def check_html_pdf_metadata(cls, values):
        """
        Check that html_data is set if content_type is HTML, or pdf_data is set if content_type is PDF.

        TODO: this is copied from `ParserOutput` in the document parser. Do we want to move it to a common place so both repos can use it?
        """
        if (
            values["document_content_type"] == CONTENT_TYPE_HTML
            and values["html_data"] is None
        ):
            raise ValueError("html_metadata must be set for HTML documents")

        if (
            values["document_content_type"] == CONTENT_TYPE_PDF
            and values["pdf_data"] is None
        ):
            raise ValueError("pdf_metadata must be null for HTML documents")

        if values["document_content_type"] not in {
            CONTENT_TYPE_HTML,
            CONTENT_TYPE_PDF,
        } and (values["html_data"] is not None or values["pdf_data"] is not None):
            raise ValueError(
                "html_metadata and pdf_metadata must be null for documents an "
                "unsupported content type."
            )

        return values
