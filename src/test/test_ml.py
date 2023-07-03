import numpy as np

from src import config
from src.ml import SBERTEncoder


def test_encoder():
    """Assert that we can instantiate an encoder object and encode textual data using the class methods."""

    encoder = SBERTEncoder(config.SBERT_MODEL)

    assert encoder is not None

    assert isinstance(encoder.encode("Hello world!"), np.ndarray)

    assert isinstance(encoder.encode_batch(["Hello world!"]*100), np.ndarray)

    assert encoder.dimension == 768
