from sentence_transformers import SentenceTransformer

from src import config

if __name__ == '__main__':
    SentenceTransformer(
        config.SBERT_MODEL, cache_folder=config.INDEX_ENCODER_CACHE_FOLDER
    )
