from pathlib import Path
import tempfile
from unittest import mock

from click.testing import CliRunner
from cloudpathlib.local import LocalS3Path
import pytest
import numpy as np

from cli.text2embeddings import main as cli_main


@pytest.fixture()
def test_input_dir() -> Path:
    return (Path(__file__).parent / "test_data" / "input").resolve()


def test_run_encoder_local(test_input_dir: Path):
    """Test that the encoder runs with local input and output paths and outputs the correct files."""

    with tempfile.TemporaryDirectory() as output_dir:
        runner = CliRunner()
        result = runner.invoke(cli_main, [str(test_input_dir), output_dir])
        assert result.exit_code == 0

        assert (Path(output_dir) / "test_html.npy").exists()
        assert (Path(output_dir) / "test_pdf.npy").exists()


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
        assert (LocalS3Path(output_dir) / "test_html.npy").exists()


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
