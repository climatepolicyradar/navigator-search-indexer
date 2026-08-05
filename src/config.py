"""In-app config. Set by environment variables."""

import os
from typing import Set


class ConfigError(Exception):
    """Configuration Error."""

    pass


# General config
# Language names as the Snowflake export writes them, not ISO codes.
TARGET_LANGUAGES: Set[str] = {"en"}
ENCODER_SUPPORTED_LANGUAGES: Set[str] = {"English"}

# Vespa config
VESPA_CONNECTIONS: int = int(os.getenv("VESPA_CONNECTIONS", "100"))
VESPA_DOCUMENT_BATCH_SIZE: int = int(os.getenv("VESPA_BATCH_SIZE", "10000"))
VESPA_INSTANCE_URL: str = os.getenv("VESPA_INSTANCE_URL", "")
VESPA_CERT_LOCATION: str = os.getenv("VESPA_CERT_LOCATION", "")
VESPA_KEY_LOCATION: str = os.getenv("VESPA_KEY_LOCATION", "")
VESPA_NAMESPACE_PREFIX: str = os.getenv("VESPA_NAMESPACE_PREFIX", "navigator")
DEVELOPMENT_MODE: bool = os.getenv("DEVELOPMENT_MODE", "False").lower() == "true"
