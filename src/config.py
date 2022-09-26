"""In-app config. Set by environment variables."""

import os

CDN_URL: str = os.getenv("CDN_URL", "https://cdn.climatepolicyradar.org")
