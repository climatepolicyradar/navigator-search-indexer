import os

from click.testing import CliRunner

from cli.index_data import run_as_cli


def test_integration():
    fixture_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s3_fixtures")
    
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
