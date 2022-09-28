from pathlib import Path
import tempfile
from unittest import mock

from click.testing import CliRunner
from cloudpathlib.local import LocalS3Path
import pytest
import numpy as np

from cli.text2embeddings import main as cli_main
from src.base import IndexerInput


@pytest.fixture()
def test_input_dir() -> Path:
    return (Path(__file__).parent / "test_data" / "text2embeddings_input").resolve()


@pytest.fixture()
def test_input_dir_bad_data() -> Path:
    return (Path(__file__).parent / "test_data" / "text2embeddings_input_bad").resolve()


def test_run_encoder_local(test_input_dir: Path):
    """Test that the encoder runs with local input and output paths and outputs the correct files."""

    with tempfile.TemporaryDirectory() as output_dir:
        runner = CliRunner()
        result = runner.invoke(cli_main, [str(test_input_dir), output_dir])
        assert result.exit_code == 0

        assert set(Path(output_dir).glob("*.json")) == {
            Path(output_dir) / "test_html.json",
            Path(output_dir) / "test_pdf.json",
        }
        assert set(Path(output_dir).glob("*.npy")) == {
            Path(output_dir) / "test_html.npy",
            Path(output_dir) / "test_pdf.npy",
        }

        for path in Path(output_dir).glob("*.json"):
            assert IndexerInput.parse_raw(path.read_text())

        for path in Path(output_dir).glob("*.npy"):
            assert np.load(path).shape[1] == 768


def test_run_encoder_local_fail_bad_input(test_input_dir_bad_data: Path):
    """Test that the encoder fails with bad input data."""

    with tempfile.TemporaryDirectory() as output_dir:
        runner = CliRunner()
        result = runner.invoke(cli_main, [str(test_input_dir_bad_data), output_dir])
        assert result.exit_code == 1


def test_run_encoder_s3(test_input_dir: Path):
    """Test that the encoder runs with S3 input and output paths and outputs the correct files."""

    input_dir = "s3://test-bucket/test-input-dir"
    output_dir = "s3://test-bucket/test-output-dir"

    # Copy test data to mock of S3 path
    input_file_path = LocalS3Path(f"{input_dir}/test_html.json")
    input_file_path.write_text((test_input_dir / "test_html.json").read_text())

    with mock.patch("cli.text2embeddings.S3Path", LocalS3Path):
        runner = CliRunner()
        result = runner.invoke(cli_main, [input_dir, output_dir, "--s3"])

        assert result.exit_code == 0

        assert set(LocalS3Path(output_dir).iterdir()) == {
            LocalS3Path(f"{output_dir}/test_html.json"),
            LocalS3Path(f"{output_dir}/test_html.npy"),
        }


def test_run_parser_skip_already_done(caplog, test_input_dir) -> None:
    """Test that files which have already been parsed are skipped by default."""

    with tempfile.TemporaryDirectory() as output_dir:
        for fname_to_skip in ("test_pdf", "test_html"):
            with open(Path(output_dir) / f"{fname_to_skip}.npy", "wb") as f:
                f.write(np.random.rand(768).astype(np.float32).tobytes())

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                str(test_input_dir),
                output_dir,
            ],
        )

        assert result.exit_code == 0

        assert (
            "Found 2 documents that have already been encoded. Skipping."
            in caplog.messages
        )

        assert "No more documents to encode. Exiting." in caplog.messages
