from email.generator import Generator
from cloudpathlib import S3Path 

from src.index.opensearch import get_text_document_generator
from cpr_data_access.pipeline_general_models import CONTENT_TYPE_PDF

def test_get_text_document_generator(
        test_stepfunctions_client,
        pipeline_s3_client,
        test_document_data,
        embeddings_dir_as_path: S3Path,
        caplog
    ) -> None:
    """
    Test that the generator successfully represents json files.
    
    Particularly page numbers. 
    """
    # TODO create list of parser outputs for tasks 
    # TODO create a mock s3 path with some numpy files 
    text_document_generator = get_text_document_generator(
        tasks=[test_document_data],
        embedding_dir_as_path=embeddings_dir_as_path,
        translated=False,
        content_types=[CONTENT_TYPE_PDF],
    )
    
    
    assert isinstance(text_document_generator, Generator)
    
    # TODO Test that we successfully filter for translated 
    # TODO test that we correctly filter for content-type
    # TODO Test that we successfully remove the correct block types 
    # TODO Test that we successfully preserve page numbers (or +1 if that's the desired functionality)
