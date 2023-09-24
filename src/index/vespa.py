import logging
from pathlib import Path

_LOGGER = logging.getLogger(__name__)


class VespaIndex:
    """Load data into a Vespa index."""
    # TODO: Implement

    def __init__(
        self,
        url: str,
        key: Path,
        cert: Path,
        namespace: str,
    ):
        self._url = url
        self._key = key
        self._cert = cert
        self._namespace = namespace

        raise NotImplementedError("Vespa Indexing not yet implemented")
