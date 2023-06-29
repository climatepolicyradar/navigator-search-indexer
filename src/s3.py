import json
from typing import Any

import boto3
from botocore.exceptions import ClientError
from aws_error_utils import errors

from src.config import S3_PATTERN


def validate_s3_pattern(s3_path: str):
    """Validates that a string is a valid s3 path."""
    s3_match = S3_PATTERN.match(s3_path)
    if s3_match is None:
        raise Exception(f"Key does not represent an s3 path: {s3_path}")
    bucket = s3_match.group("bucket")
    key = s3_match.group("prefix")
    s3client = boto3.client("s3")
    return bucket, key, s3client


# TODO do we want to instantiate one client object and pass that through rather than instantiating each time?
def check_file_exists_in_s3(s3_path: str):
    """Checks whether a file exists in an S3 bucket."""
    bucket, key, s3client = validate_s3_pattern(s3_path)
    try:
        s3client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False
    except errors.NoSuchBucket:
        return False
    except errors.NoSuchKey:
        return False
    except Exception as e:
        raise e


def get_s3_keys_with_prefix(s3_prefix: str) -> list[str]:
    """
    Get a list of keys in an S3 bucket with a given prefix. Returns an empty list if the prefix does not exist
    or is empty.

    We use this instead of cloudpathlib's glob because it's much faster. Relevant issue:
    https://github.com/drivendataorg/cloudpathlib/issues/274.

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


def s3_object_read_text(s3_path: str) -> str:
    """
    Read text from an S3 object.

    :param s3_path: path to S3 object, including s3:// prefix
    :return str: text of S3 object
    """
    bucket, key, s3client = validate_s3_pattern(s3_path)

    try:
        response = s3client.get_object(Bucket=bucket, Key=key)
    except errors.NoSuchBucket:
        raise ValueError(f"Bucket {bucket} does not exist")
    except errors.NoSuchKey:
        raise ValueError(f"Key {key} does not exist")
    except Exception as e:
        raise e

    return response["Body"].read().decode("utf-8")


def write_json_to_s3(json_data: dict, s3_path: str) -> None:
    """Writes JSON data to an S3 bucket."""
    bucket, key, s3client = validate_s3_pattern(s3_path)

    # Upload the JSON string to S3
    try:
        s3client.put_object(Body=json_data, Bucket=bucket, Key=key)
    except errors.NoSuchBucket:
        raise ValueError(f"Bucket {bucket} does not exist")
    except Exception as e:
        raise e


def save_ndarray_to_s3_as_npy(array: Any, s3_path: str) -> None:
    """Saves a NumPy ndarray to an S3 bucket as a .npy file."""
    bucket, key, s3client = validate_s3_pattern(s3_path)

    # Convert the NumPy array to bytes
    array_bytes = array.tobytes()

    # Upload the array bytes to S3
    try:
        s3client.put_object(Body=array_bytes, Bucket=bucket, Key=key)
    except errors.NoSuchBucket:
        raise ValueError(f"Bucket {bucket} does not exist")
    except Exception as e:
        raise e
