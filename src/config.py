"""In-app config. Set by environment variables."""

import os


class ConfigError(Exception):
    """Configuration Error."""

    pass


# General config
# NOTE: Comparable list being maintained at https://github.com/climatepolicyradar/knowledge-graph/blob/83bda1919cea415b6fc0813bfb214a23662a060b/flows/inference.py#L29-L33
BLOCKS_TO_FILTER = os.getenv("BLOCKS_TO_FILTER", "Table,Figure").split(",")

# Vespa config
VESPA_CONNECTIONS: int = int(os.getenv("VESPA_CONNECTIONS", "100"))
VESPA_DOCUMENT_BATCH_SIZE: int = int(os.getenv("VESPA_BATCH_SIZE", "10000"))
VESPA_INSTANCE_URL: str = os.getenv("VESPA_INSTANCE_URL", "")
VESPA_CERT_LOCATION: str = os.getenv("VESPA_CERT_LOCATION", "")
VESPA_KEY_LOCATION: str = os.getenv("VESPA_KEY_LOCATION", "")
VESPA_NAMESPACE_PREFIX: str = os.getenv("VESPA_NAMESPACE_PREFIX", "navigator")
DEVELOPMENT_MODE: bool = os.getenv("DEVELOPMENT_MODE", "False").lower() == "true"
