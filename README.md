# Vespa Indexer

The code in this folder contain a CLI tool used to index data into the Navigator search index:

* `index_data.py`: loads document metadata from the Navigator database and indexes this data alongside the text and embeddings created from embeddings generation into the search index.

# Vespa test setup

```
make build
make vespa_setup
make test
```
