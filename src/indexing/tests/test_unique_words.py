from indexing.preprocessing import fetch_and_tokenize_documents
from indexing.inverted_index import fetch_inverted_index
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
results_folder = "results"
inverted_index_file = os.path.join(results_folder, "inverted_index.json")

def test_unique_words_count():
    tokenized_docs = fetch_and_tokenize_documents()
    unique_words = set()
    for tokens in tokenized_docs:
        unique_words.update(tokens)
    num_unique_words = len(unique_words)
    logger.info(f"Number of unique words across all documents: {num_unique_words}")

    inverted_index = fetch_inverted_index(inverted_index_file)
    num_inverted_index_keys = len(inverted_index)
    logger.info(f"Number of unique words in the inverted index: {num_inverted_index_keys}")

    assert num_unique_words == num_inverted_index_keys, (
        f"Mismatch in unique words count: {num_unique_words} (docs) vs {num_inverted_index_keys} (index)"
    )
    logger.info("Test passed: The number of unique words in the documents matches the number of keys in the inverted index.")

test_unique_words_count()