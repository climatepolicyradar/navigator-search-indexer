import os

from click.testing import CliRunner
import pytest

from cli.index_data import run_as_cli


@pytest.mark.usefixtures("cleanup_test_vespa_after")
def test_integration():
    fixture_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "s3_files")
    
    runner = CliRunner()
    result = runner.invoke(
        run_as_cli,
        args = [
            fixture_dir,
            "--index-type",
            "vespa",
        ],
    )

    assert result.exit_code == 0, (
        f"Exception: {result.exception if result.exception else None}\n"
        f"Stdout: {result.stdout}"
    )
