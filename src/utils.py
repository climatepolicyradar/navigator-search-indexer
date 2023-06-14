import json
import logging
import os
from typing import Any

import boto3
from aws_error_utils.aws_error_utils import errors


from src.base import IndexerInput, TextBlock, BlockTypes
from src.config import S3_PATTERN

logger = logging.getLogger(__name__)


def replace_text_blocks(block: IndexerInput, new_text_blocks: list[TextBlock]):
    """Updates the text blocks in the IndexerInput object."""
    if block.pdf_data is not None:
        block.pdf_data.text_blocks = new_text_blocks
    elif block.html_data is not None:
        block.html_data.text_blocks = new_text_blocks

    return block


def filter_blocks(indexer_input: IndexerInput, remove_block_types: list[str]) -> list[TextBlock]:
    """Given an Indexer Input filter the contained TextBlocks and return this as a list of TextBlocks."""
    filtered_blocks = []
    for block in indexer_input.get_text_blocks(including_invalid_html=True):
        if block.type.title() not in remove_block_types:
            filtered_blocks.append(block)
        else:
            logger.info(
                f"Filtered {block.type} block from {indexer_input.document_id}.",
                extra={
                    "props": {
                        "document_id": indexer_input.document_id,
                        "block_type": block.type,
                        "remove_block_types": remove_block_types
                    }
                }
            )
    return filtered_blocks


def filter_on_block_type(inputs: list[IndexerInput], remove_block_types: list[str]) -> list[IndexerInput]:
    """Filter a sequence of IndexerInputs to remove the textblocks that are of the types declared in the remove block
    types array."""
    for _filter in remove_block_types:
        try:
            BlockTypes(_filter)
        except NameError:
            logger.warning(f"Blocks to filter should be of a known block type, removing {_filter} from the list.")
            remove_block_types.remove(_filter)

    return [
        replace_text_blocks(
            block=_input,
            new_text_blocks=filter_blocks(indexer_input=_input, remove_block_types=remove_block_types)
        )
        for _input in inputs
    ]


# TODO do we want to instantiate one client object and pass that through rather than reinstatiating each time?
def _get_s3_keys_with_prefix(s3_prefix: str) -> list[str]:
    """
    Get a list of keys in an S3 bucket with a given prefix. Returns an empty list if the prefix does not exist or is empty.

    We use this instead of cloudpathlib's glob because it's much faster. Relevant issue: https://github.com/drivendataorg/cloudpathlib/issues/274.

    :param s3_prefix: prefix, including s3:// at the start
    :raises Exception: if prefix does not represent an s3 path
    :return list[str]: list of full paths to objects in bucket, excluding s3:// prefix
    """
    s3_match = S3_PATTERN.match(s3_prefix)
    if s3_match is None:
        raise Exception(f"Prefix does not represent an s3 path: {s3_prefix}")

    bucket = s3_match.group("bucket")
    prefix = s3_match.group("prefix").rstrip("/") + "/"
    s3client = boto3.client("s3")

    try:
        list_response = s3client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    except errors.NoSuchBucket:
        raise ValueError(f"Bucket {bucket} does not exist")
    except Exception as e:
        raise e

    files = [o["Key"] for o in list_response.get("Contents", []) if o["Key"] != prefix]

    finished_listing = not list_response["IsTruncated"]
    while not finished_listing:
        continuation_token = list_response.get("NextContinuationToken")
        list_response = s3client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            ContinuationToken=continuation_token,
        )
        files.extend(
            [o["Key"] for o in list_response["Contents"] if o["Key"] != prefix]
        )
        finished_listing = not list_response["IsTruncated"]

    return files


def _s3_object_read_text(s3_path: str) -> str:
    """
    Read text from an S3 object.

    :param s3_key: path to S3 object, including s3:// prefix
    :return str: text of S3 object
    """
    # TODO de-duplicate code repetition
    s3_match = S3_PATTERN.match(s3_path)
    if s3_match is None:
        raise Exception(f"Key does not represent an s3 path: {s3_path}")

    bucket = s3_match.group("bucket")
    key = s3_match.group("prefix")
    s3client = boto3.client("s3")

    try:
        response = s3client.get_object(Bucket=bucket, Key=key)
    except errors.NoSuchBucket:
        raise ValueError(f"Bucket {bucket} does not exist")
    except errors.NoSuchKey:
        raise ValueError(f"Key {key} does not exist")
    except Exception as e:
        raise e

    return response["Body"].read().decode("utf-8")


def _write_json_to_s3(json_data: dict, s3_path: str) -> None:
    """Writes JSON data to an S3 bucket."""
    s3_match = S3_PATTERN.match(s3_path)
    if s3_match is None:
        raise Exception(f"Key does not represent an s3 path: {s3_path}")

    bucket = s3_match.group("bucket")
    key = s3_match.group("prefix")
    s3client = boto3.client("s3")

    # Convert JSON data to a string
    json_string = json.dumps(json_data)

    # Upload the JSON string to S3
    try:
        s3client.put_object(Body=json_string, Bucket=bucket, Key=key)
    except errors.NoSuchBucket:
        raise ValueError(f"Bucket {bucket} does not exist")
    except Exception as e:
        raise e


def _save_ndarray_to_s3_as_npy(array: Any, s3_path: str) -> None:
    """Saves a NumPy ndarray to an S3 bucket as a .npy file."""
    s3_match = S3_PATTERN.match(s3_path)
    if s3_match is None:
        raise Exception(f"Key does not represent an s3 path: {s3_path}")

    bucket = s3_match.group("bucket")
    key = s3_match.group("prefix")
    s3client = boto3.client("s3")

    # Convert the NumPy array to bytes
    array_bytes = array.tobytes()

    # Upload the array bytes to S3
    try:
        s3client.put_object(Body=array_bytes, Bucket=bucket, Key=key)
    except errors.NoSuchBucket:
        raise ValueError(f"Bucket {bucket} does not exist")
    except Exception as e:
        raise e


def _check_file_exists_in_s3(s3_path: str):
    """Checks whether a file exists in an S3 bucket."""
    s3_match = S3_PATTERN.match(s3_path)
    if s3_match is None:
        raise Exception(f"Key does not represent an s3 path: {s3_path}")

    bucket = s3_match.group("bucket")
    key = s3_match.group("prefix")
    s3client = boto3.client("s3")

    try:
        s3client.head_object(Bucket=bucket, Key=key)
        return True
    except errors.NoSuchBucket or errors.NoSuchKey:
        return False
    except Exception as e:
        raise e


def _get_ids_with_suffix(files: list[str], suffix: str) -> set[str]:
    """Get a set of the ids of the files with the given suffix."""
    files = [file for file in files if file.endswith(suffix)]
    return set([os.path.splitext(os.path.basename(file))[0] for file in files])
