import io
import json
import logging
import tempfile
from pathlib import Path

import numpy as np
from click.testing import CliRunner

from cli.text2embeddings import run_as_cli
from cpr_data_access.parser_models import ParserOutput


def test_run_encoder_local(
    test_html_file_json,
    test_pdf_file_json,
    test_no_content_type_file_json,
):
    """Test that the encoder runs with local input and output paths and outputs the correct files."""

    with tempfile.TemporaryDirectory() as input_dir:
        with tempfile.TemporaryDirectory() as output_dir:
            # Create test files
            for file in [
                test_html_file_json,
                test_pdf_file_json,
                test_no_content_type_file_json,
            ]:
                file_path = Path(input_dir) / f"{file['document_id']}.json"
                file_path.write_text(json.dumps(file))

            runner = CliRunner()
            result = runner.invoke(run_as_cli, [input_dir, output_dir])
            assert result.exit_code == 0

            assert set(Path(output_dir).glob("*.json")) == {
                Path(output_dir) / "test_html.json",
                Path(output_dir) / "test_pdf.json",
                Path(output_dir) / "test_no_content_type.json",
            }
            assert set(Path(output_dir).glob("*.npy")) == {
                Path(output_dir) / "test_html.npy",
                Path(output_dir) / "test_pdf.npy",
                Path(output_dir) / "test_no_content_type.npy",
            }

            for path in Path(output_dir).glob("*.json"):
                assert ParserOutput.parse_obj(json.loads(path.read_text()))

            for path in Path(output_dir).glob("*.npy"):
                assert np.load(str(path)).shape[1] == 768

            # test_html has the `has_valid_text` flag set to false, so the numpy file
            # should only contain a description embedding
            assert np.load(str(Path(output_dir) / "test_html.npy")).shape == (1, 768)


def test_s3_client(
    s3_bucket_and_region,
    pipeline_s3_objects_main,
    pipeline_s3_client_main,
    input_prefix,
):
    """Prior to running the embeddings generation tests assert that the mock s3 bucket is in the required state."""
    list_response = pipeline_s3_client_main.client.list_objects_v2(
        Bucket=s3_bucket_and_region["bucket"], Prefix=input_prefix
    )
    assert list_response["KeyCount"] == len(pipeline_s3_objects_main)


def test_run_encoder_s3(
    s3_bucket_and_region,
    pipeline_s3_objects_main,
    pipeline_s3_client_main,
    test_input_dir_s3,
    test_output_dir_s3,
    output_prefix,
):
    """Test that the encoder runs with S3 input and output paths and outputs the correct files."""

    runner = CliRunner()
    result = runner.invoke(run_as_cli, [test_input_dir_s3, test_output_dir_s3, "--s3"])

    assert result.exit_code == 0

    list_response = pipeline_s3_client_main.client.list_objects_v2(
        Bucket=s3_bucket_and_region["bucket"], Prefix=output_prefix
    )

    assert list_response["KeyCount"] == len(pipeline_s3_objects_main) * 2

    files = [
        o["Key"] for o in list_response.get("Contents", []) if o["Key"] != output_prefix
    ]

    assert len(files) == len(pipeline_s3_objects_main) * 2

    assert set(files) == {
        f"{output_prefix}/test_html.json",
        f"{output_prefix}/test_html.npy",
        f"{output_prefix}/test_no_content_type.json",
        f"{output_prefix}/test_no_content_type.npy",
        f"{output_prefix}/test_pdf.json",
        f"{output_prefix}/test_pdf.npy",
    }

    s3_files_json = [file for file in files if file.endswith(".json")]
    s3_files_npy = [file for file in files if file.endswith(".npy")]

    assert len(s3_files_json) == len(pipeline_s3_objects_main)
    assert len(s3_files_npy) == len(pipeline_s3_objects_main)

    for file in s3_files_json:
        file_obj = pipeline_s3_client_main.client.get_object(
            Bucket=s3_bucket_and_region["bucket"], Key=file
        )
        file_text = file_obj["Body"].read().decode("utf-8")
        file_json = json.loads(file_text)
        assert ParserOutput.parse_obj(file_json)

    for file in s3_files_npy:
        file_obj = pipeline_s3_client_main.client.get_object(
            Bucket=s3_bucket_and_region["bucket"], Key=file
        )
        file_text = file_obj["Body"]
        file_bytes = io.BytesIO(file_text.read())
        assert np.load(file_bytes).shape[1] == 768


def test_run_parser_skip_already_done(
    test_html_file_json, test_pdf_file_json, test_no_content_type_file_json, caplog
) -> None:
    """Test that files which have already been parsed are skipped by default."""

    with tempfile.TemporaryDirectory() as input_dir:
        with tempfile.TemporaryDirectory() as output_dir:
            # Create test files in input and output directories
            for file in [
                test_html_file_json,
                test_pdf_file_json,
                test_no_content_type_file_json,
            ]:
                file_path = Path(input_dir) / f"{file['document_id']}.json"
                file_path.write_text(json.dumps(file))

                file_path = Path(output_dir) / f"{file['document_id']}.npy"
                file_path.write_text(str(np.array([1, 2, 3]).tobytes()))

            caplog.set_level(logging.INFO)
            runner = CliRunner()
            result = runner.invoke(
                run_as_cli,
                [
                    input_dir,
                    output_dir,
                ],
            )

            assert result.exit_code == 0

            all_messages = ""
            for i in caplog.messages:
                all_messages = all_messages + i

            assert (
                "3 documents found that have already been encoded. Skipping."
                in all_messages
            )

            assert "No more documents to encode. Exiting." in all_messages
