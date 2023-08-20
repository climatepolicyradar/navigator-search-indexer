import logging
import os
from pathlib import Path
from time import sleep
from typing import Generator, Optional, Iterable, Sequence, Tuple

from opensearchpy import ConnectionTimeout, OpenSearch, helpers
from tqdm.auto import tqdm
import requests

from src import config
from src.index_mapping import ALL_FIELDS

_LOGGER = logging.getLogger(__name__)


class OpenSearchIndex:
    """
    Useful methods on an Opensearch index.

    - Connect to OpenSearch instance
    - Define an index mapping
    - Load data into an index
    """

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

        mapping[ALL_FIELDS["id"][0]] = {
            "type": "keyword",
            "normalizer": "folding",
            # Load ordinals on indexing for this field for faster aggregations.
            "eager_global_ordinals": True,
        }

        for field in ALL_FIELDS["sortable"]:
            mapping[field] = {
                "type": "keyword",
                "normalizer": "folding",
            }

        for field in ALL_FIELDS["date"]:
            mapping[field] = {"type": "date", "format": "dd/MM/yyyy"}

        for field in ALL_FIELDS["integer"]:
            mapping[field] = {"type": "integer"}

        for field in ALL_FIELDS["searchable"]:
            mapping[field] = {
                "type": "text",
                "analyzer": "folding",
            }

        for field in ALL_FIELDS["embedding"]:
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

        for field in ALL_FIELDS["boolean"]:
            mapping[field] = {"type": "boolean"}

        for field in ALL_FIELDS["categorical"]:
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
                    # This analyser folds non-ASCII characters into ASCII equivalents,
                    # but preserves the original.
                    # E.g. a search for "é" will return results for "e" and "é".
                    "analyzer": {
                        "folding": {
                            "tokenizer": "standard",
                            "filter": ["lowercase", "ascii_folding_preserve_original"],
                        }
                    },
                    # This normalizer does the same as the folding analyser, but is
                    # used for keyword fields.
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

    def delete_index(self):
        """Delete the current index"""
        delete_attempt_count = 0
        delete_succeeded = False
        while delete_attempt_count < 5 and not delete_succeeded:
            try:
                self.opns.indices.delete(
                    index=self.index_name,
                    ignore=[404],
                    request_timeout=config.OPENSEARCH_BULK_REQUEST_TIMEOUT,
                )
                delete_succeeded = True
            except ConnectionTimeout:
                delete_attempt_count += 1
                sleep(5 * delete_attempt_count)
        if not delete_succeeded:
            raise RuntimeError(
                f"Failed to delete existing index '{self.index_name}' after "
                f"{delete_attempt_count} attempts"
            )

    def create_index(self, n_replicas: int):
        """Create the index this object is configured to interqact with"""
        create_attempt_count = 0
        create_succeeded = False
        while create_attempt_count < 5 and not create_succeeded:
            try:
                self.opns.indices.create(
                    index=self.index_name,
                    body=self._index_body(n_replicas),
                    request_timeout=config.OPENSEARCH_BULK_REQUEST_TIMEOUT,
                )
                create_succeeded = True
            except ConnectionTimeout:
                create_attempt_count += 1
                sleep(5)
        if not create_succeeded:
            raise RuntimeError(
                f"Failed to create index '{self.index_name}' after "
                f"{create_attempt_count} attempts"
            )

    def set_index_refresh_interval(self, interval: int, timeout: int = 10):
        """
        Set the refresh interval (seconds) for the index.

        If interval=-1, refresh is disabled.
        """

        interval_seconds = interval if interval == -1 else f"{interval}s"
        timeout_seconds = f"{timeout}s"

        self.opns.indices.put_settings(
            index=self.index_name,
            body={"index.refresh_interval": interval_seconds},
            timeout=timeout_seconds,
        )

    def bulk_index(self, actions: Iterable[dict]):
        """Bulk load data into the index.

        # TODO: in future, we may want to expose `streaming_bulk` kwargs to allow
        for more control over the bulk load.

        Args:
            actions (Iterable[dict]): a list of documents or actions to be indexed.
        """

        actions = tqdm(actions, unit="docs")
        batch_successes = 0
        batch_failures = 0

        for ok, info in helpers.streaming_bulk(
            client=self.opns,
            index=self.index_name,
            actions=actions,
            request_timeout=config.OPENSEARCH_BULK_REQUEST_TIMEOUT,
            max_retries=10,  # Hardcoded for now as purpose to avoid HTTP/429
            initial_backoff=10,
            chunk_size=200,
            max_chunk_bytes=20 * 1000 * 1000,
        ):
            if ok:
                batch_successes += 1
            else:
                batch_failures += 1
                _LOGGER.error(f"Failed to process batch: '{info}'")

        _LOGGER.info(f"Processed {batch_successes} batch(es) successfully")
        _LOGGER.info(f"Processed {batch_failures} batch(es) unsuccessfully")

        if batch_failures:
            raise RuntimeError(
                f"Failed to process {batch_failures} batch(es) during index generation"
            )

    def warmup_knn(self) -> bool:
        """Load the KNN index into memory by calling the index warmup API.

        Returns when the warmup is complete, or returns False and logs the error
        message if it fails.

        Returns:
            bool: whether the warmup request succeeded
        """

        url = f"{self._url}/_plugins/_knn/warmup/{self.index_name}?pretty"

        response = requests.get(
            url,
            auth=self._login,  # type: ignore
        )

        if response.status_code == 200:
            return True
        else:
            _LOGGER.warning(
                "KNN index warmup API call returned non-200 status code. "
                f"Full response {response.json()}"
            )
            return False

    def set_num_replicas(self, n_replicas: int):
        """Set the number of replicas for the index.

        Args:
            n_replicas (int): number of replicas to create for the index.
        """
        self.opns.indices.put_settings(
            index=self.index_name,
            body={"index.number_of_replicas": n_replicas},
        )


def populate_and_warmup_index(
    doc_generator: Generator[dict, None, None], index_name: str
):
    """
    Load documents into an Opensearch index.

    This function also loads the KNN index into native memory (warmup).

    :param doc_generator: generator of Opensearch documents to index
    :param index_name: name of index to load documents into
    """

    _LOGGER.info(f"Loading documents into index {index_name}")

    opensearch = OpenSearchIndex(
        url=os.environ["OPENSEARCH_URL"],
        username=os.environ["OPENSEARCH_USER"],
        password=os.environ["OPENSEARCH_PASSWORD"],
        index_name=index_name,
        opensearch_connector_kwargs={
            "use_ssl": config.OPENSEARCH_USE_SSL,
            "verify_certs": config.OPENSEARCH_VERIFY_CERTS,
            "ssl_show_warn": config.OPENSEARCH_SSL_SHOW_WARN,
        },
        embedding_dim=config.OPENSEARCH_INDEX_EMBEDDING_DIM,
    )

    # Disabling replicas during indexing means that the KNN index is copied to
    # replicas after indexing is complete rather than multiple, potentially
    # different KNN indices being created in parallel.
    # It should also speed up indexing.
    opensearch.create_index(n_replicas=0)

    # We disable index refreshes during indexing to speed up the indexing process,
    # and to ensure only 1 segment is created per shard. This also speeds up KNN
    # queries and aggregations according to the Opensearch and Elasticsearch docs.
    opensearch.set_index_refresh_interval(-1, timeout=60)
    opensearch.bulk_index(actions=doc_generator)
    opensearch.set_num_replicas(config.OPENSEARCH_INDEX_NUM_REPLICAS)

    # TODO: we wrap this in a try/except block because for now because sometimes
    # it times out, and we don't want the whole >1hr indexing process to fail if
    # this happens. We should stop doing this if we ever care what the refresh
    # interval is, i.e. when we plan on incrementally adding data to the index.
    try:
        # 1 second refresh interval is the Opensearch default
        opensearch.set_index_refresh_interval(1, timeout=60)
    except Exception as e:
        _LOGGER.info(f"Failed to set index refresh interval after indexing: {e}")

    opensearch.warmup_knn()


def delete_index(index_name: str):
    """
    Deletes the index.

    :param index_name: name of index to delete
    """

    _LOGGER.info(f"Deleting index {index_name}")

    opensearch = OpenSearchIndex(
        url=os.environ["OPENSEARCH_URL"],
        username=os.environ["OPENSEARCH_USER"],
        password=os.environ["OPENSEARCH_PASSWORD"],
        index_name=index_name,
        opensearch_connector_kwargs={
            "use_ssl": config.OPENSEARCH_USE_SSL,
            "verify_certs": config.OPENSEARCH_VERIFY_CERTS,
            "ssl_show_warn": config.OPENSEARCH_SSL_SHOW_WARN,
        },
        embedding_dim=config.OPENSEARCH_INDEX_EMBEDDING_DIM,
    )
    opensearch.delete_index()


def populate_opensearch(
    indices_to_populate: Sequence[Tuple[str, Generator[dict, None, None]]]
):
    """Populate all indices in the given list"""
    # First remove the indices
    for index_name, _ in indices_to_populate:
        delete_index(index_name)

    for index_name, doc_generator in indices_to_populate:
        populate_and_warmup_index(doc_generator, index_name)
