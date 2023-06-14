from pathlib import Path
import tempfile
from unittest import mock

from click.testing import CliRunner
import pytest
import numpy as np

from cli.text2embeddings import run_as_cli
from src.base import IndexerInput


def test_run_encoder_local(test_input_dir: Path):
    """Test that the encoder runs with local input and output paths and outputs the correct files."""

    with tempfile.TemporaryDirectory() as output_dir:
        runner = CliRunner()
        result = runner.invoke(run_as_cli, [str(test_input_dir), output_dir])
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
            assert IndexerInput.parse_raw(path.read_text())

        for path in Path(output_dir).glob("*.npy"):
            assert np.load(path).shape[1] == 768

        # test_html has the `has_valid_text` flag set to false, so the numpy file should only contain a
        # description embedding
        assert np.load(Path(output_dir) / "test_html.npy").shape == (1, 768)


def test_run_encoder_local_fail_bad_input(test_input_dir_bad_data: Path):
    """Test that the encoder fails with bad input data."""

    with tempfile.TemporaryDirectory() as output_dir:
        runner = CliRunner()
        result = runner.invoke(run_as_cli, [str(test_input_dir_bad_data), output_dir])
        assert result.exit_code == 1


def test_run_encoder_s3(pipeline_s3_client, test_input_dir_s3, test_output_dir_s3):
    """Test that the encoder runs with S3 input and output paths and outputs the correct files."""

    runner = CliRunner()
    result = runner.invoke(run_as_cli, [test_input_dir_s3, test_output_dir_s3, "--s3"])

    assert result.exit_code == 0

    # TODO assert output files exist
    # assert set(LocalS3Path(output_dir).iterdir()) == {
    #     LocalS3Path(f"{output_dir}/test_html.json"),
    #     LocalS3Path(f"{output_dir}/test_html.npy"),
    # }


def test_run_parser_skip_already_done(caplog, test_input_dir) -> None:
    """Test that files which have already been parsed are skipped by default."""

    with tempfile.TemporaryDirectory() as output_dir:
        for fname_to_skip in ("test_pdf", "test_html", "test_no_content_type"):
            with open(Path(output_dir) / f"{fname_to_skip}.npy", "wb") as f:
                f.write(np.random.rand(768).astype(np.float32).tobytes())

        runner = CliRunner()
        result = runner.invoke(
            run_as_cli,
            [
                str(test_input_dir),
                output_dir,
            ],
        )

        assert result.exit_code == 0

        assert (
            "Found 3 documents that have already been encoded. Skipping."
            in caplog.messages
        )

        assert "No more documents to encode. Exiting." in caplog.messages
