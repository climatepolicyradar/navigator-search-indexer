from typing import Optional, Iterable
import logging

from opensearchpy import OpenSearch, helpers
from tqdm.auto import tqdm
import requests

from src import config, index_mapping

logger = logging.getLogger(__name__)


class OpenSearchIndex:
    """Methods to connect to OpenSearch instance, define an index mapping, and load data into an index."""

    def __init__(
        self,
        embedding_dim: int,
        index_name: str,
        url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        opensearch_connector_kwargs: dict = {},
    ):
        self.index_name = index_name

        self._url = url
        self._login = (username, password)
        self._opensearch_connector_kwargs = opensearch_connector_kwargs

        self.embedding_dim = embedding_dim

        self._connect_to_opensearch()

    def _connect_to_opensearch(
        self,
    ):

        if self._url:
            if all(self._login):
                self.opns = OpenSearch(
                    [self._url],
                    http_auth=self._login,
                    **self._opensearch_connector_kwargs,
                )
            else:
                self.opns = OpenSearch([self._url], **self._opensearch_connector_kwargs)

        else:
            self.opns = OpenSearch(**self._opensearch_connector_kwargs)

    def is_connected(self) -> bool:
        """Check if we are connected to the OpenSearch instance."""
        return self.opns.ping()

    def _generate_mapping_properties(self) -> dict:
        mapping = dict()

        mapping[index_mapping.DOCUMENT_ID_FIELD] = {
            "type": "keyword",
            "normalizer": "folding",
            # Load ordinals on indexing for this field for faster aggregations.
            "eager_global_ordinals": True,
        }

        for field in index_mapping.SORTABLE_TEXT_FIELDS:
            mapping[field] = {
                "type": "keyword",
                "normalizer": "folding",
            }

        for field in index_mapping.DATE_FIELDS:
            mapping[field] = {"type": "date", "format": "dd/MM/yyyy"}

        for field in index_mapping.INTEGER_FIELDS:
            mapping[field] = {"type": "integer"}

        for field in index_mapping.SEARCHABLE_TEXT_FIELDS:
            mapping[field] = {
                "type": "text",
                "analyzer": "folding",
            }

        for field in index_mapping.EMBEDDING_FIELDS:
            mapping[field] = {
                "type": "knn_vector",
                "dimension": config.OPENSEARCH_INDEX_EMBEDDING_DIM,
                "method": {
                    "name": "hnsw",
                    "space_type": "innerproduct",
                    "engine": "nmslib",
                    "parameters": {
                        "ef_construction": config.NMSLIB_EF_CONSTRUCTION,
                        "m": config.NMSLIB_M,
                    },
                },
            }

        for field in index_mapping.BOOLEAN_FIELDS:
            mapping[field] = {"type": "boolean"}

        for field in index_mapping.CATEGORICAL_FIELDS:
            mapping[field] = {"type": "keyword"}

        return mapping

    def _index_body(self, n_replicas: int) -> dict:
        """Define policy index fields and types"""

        return {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": config.KNN_PARAM_EF_SEARCH,
                    "number_of_shards": config.OPENSEARCH_INDEX_NUM_SHARDS,
                    "number_of_replicas": n_replicas,
                },
                "analysis": {
                    "filter": {
                        "ascii_folding_preserve_original": {
                            "type": "asciifolding",
                            "preserve_original": True,
                        }
                    },
                    # This analyser folds non-ASCII characters into ASCII equivalents, but preserves the original.
                    # E.g. a search for "é" will return results for "e" and "é".
                    "analyzer": {
                        "folding": {
                            "tokenizer": "standard",
                            "filter": ["lowercase", "ascii_folding_preserve_original"],
                        }
                    },
                    # This normalizer does the same as the folding analyser, but is used for keyword fields.
                    "normalizer": {
                        "folding": {
                            "type": "custom",
                            "char_filter": [],
                            "filter": ["lowercase", "asciifolding"],
                        }
                    },
                },
            },
            "mappings": {"properties": self._generate_mapping_properties()},
        }

    def delete_and_create_index(self, n_replicas: int):
        """Create the index, deleting any existing index of the same name first.

        Args:
            n_replicas (int): number of replicas to create for the index.
        """
        self.opns.indices.delete(index=self.index_name, ignore=[400, 404])
        self.opns.indices.create(
            index=self.index_name, body=self._index_body(n_replicas)
        )

    def set_index_refresh_interval(self, interval: int, timeout: int = 10):
        """Set the refresh interval (seconds) for the index. If interval=-1, refresh is disabled."""

        interval_seconds = interval if interval == -1 else f"{interval}s"
        timeout_seconds = f"{timeout}s"

        self.opns.indices.put_settings(
            index=self.index_name,
            body={"index.refresh_interval": interval_seconds},
            timeout=timeout_seconds,
        )

    def bulk_index(self, actions: Iterable[dict]):
        """Bulk load data into the index.

        # TODO: in future, we may want to expose `streaming_bulk` kwargs to allow for more control over the bulk load.

        Args:
            actions (Iterable[dict]): a list of documents or actions to be indexed.
        """

        actions = tqdm(actions, unit="docs")
        successes = 0

        for ok, _ in helpers.streaming_bulk(
            client=self.opns, index=self.index_name, actions=actions
        ):
            successes += ok

    def warmup_knn(self) -> bool:
        """Load the KNN index into memory by calling the index warmup API.

        Returns when the warmup is complete, or returns False and logs the error message if it fails.

        Returns:
            bool: whether the warmup request succeeded
        """

        url = f"{self._url}/_plugins/_knn/warmup/{self.index_name}?pretty"

        response = requests.get(
            url,
            auth=self._login,
        )

        if response.status_code == 200:
            return True
        else:
            logger.warning(
                f"KNN index warmup API call returned non-200 status code. Full response {response.json()}"
            )
            return False
