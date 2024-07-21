import logging
import json
import os
from collections import defaultdict
import sys
from typing import Dict, List

sys.path.append(os.path.abspath("src"))
from indexing.preprocessing import fetch_and_tokenize_documents
from indexing.process_query import fetch_inverted_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

results_folder = "results"
inverted_index_file = os.path.join(results_folder, "inverted_index.json")


def build_inverted_index(
    tokenized_docs: List[Dict[str, List[str]]]
) -> Dict[str, List[int]]:
    inverted_index = defaultdict(list)

    for doc in tokenized_docs:
        doc_id = doc["id"]
        tokens = doc["tokens"]
        for token in set(
            tokens
        ):  # Using set to avoid duplicate entries for the same document
            if doc_id not in inverted_index[token]:
                inverted_index[token].append(doc_id)

    logger.debug(f"Built inverted index successfully.")
    return inverted_index


def store_inverted_index(inverted_index: Dict[str, List[int]], file_path: str):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(inverted_index, f, ensure_ascii=False, indent=4)
        logger.info(f"Inverted index stored successfully in {file_path}")
    except Exception as e:
        logger.error(f"Error storing inverted index: {e}")


def run_inverted_index_process():
    tokenized_docs_from_json = fetch_and_tokenize_documents()
    logger.info(f"Fetched {len(tokenized_docs_from_json)} tokenized documents")

    if tokenized_docs_from_json:
        inverted_index = build_inverted_index(tokenized_docs_from_json)
        logger.info(f"Built inverted index with {len(inverted_index)} terms")
        store_inverted_index(inverted_index, inverted_index_file)


if __name__ == "__main__":
    run_inverted_index_process()
    logger.info(
        f"Loaded inverted index with {len(fetch_inverted_index(inverted_index_file))} terms"
    )
