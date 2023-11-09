import pytest

from src.index.vespa_ import _get_vespa_instance, VespaConfigError
from src import config


def test_get_vespa_instance() -> None:
    """Test that the get_vespa_instance function works as expected."""

    assert config.VESPA_INSTANCE_URL == ""
    expected_error_string = (
        "Vespa instance URL must be configured using environment variable: "
        "'VESPA_INSTANCE_URL'"
    )
    with pytest.raises(VespaConfigError) as context:
        _get_vespa_instance()
    assert expected_error_string in str(context.value)

    config.VESPA_INSTANCE_URL = "https://www.example.com"
    with pytest.raises(VespaConfigError) as context:
        _get_vespa_instance()
    assert expected_error_string not in str(context.value)
