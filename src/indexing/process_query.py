import json
import os
from collections import defaultdict
from typing import Dict, List
from indexing.preprocessing import tokenize_query
import logging

logging.basicConfig(level=logging.INFO)  # Use DEBUG level for detailed logs
logger = logging.getLogger(__name__)

results_folder = "results"
inverted_index_file = os.path.join(results_folder, "inverted_index.json")
tokenized_docs_file = os.path.join(results_folder, "tokenized_docs.json")
tf_idf_file = os.path.join(results_folder, "tf_idf_values.json")


def fetch_inverted_index(file_path: str) -> Dict[str, List[int]]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            inverted_index = json.load(f)
        # Convert lists from strings to integers
        inverted_index = {k: list(map(int, v)) for k, v in inverted_index.items()}
        return inverted_index
    except Exception as e:
        logger.error(f"Error fetching inverted index: {e}")
        return {}


def fetch_tf_idf_values(file_path: str) -> Dict[int, Dict[str, float]]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # Convert document IDs from strings to integers
        tf_idf_values = {
            int(doc_id): token_scores for doc_id, token_scores in raw_data.items()
        }
        return tf_idf_values
    except Exception as e:
        logger.error(f"Error loading TF-IDF values from {file_path}: {e}")
        return {}


def retrieve_documents_from_index(
    inverted_index: Dict[str, List[int]], query_tokens: List[str]
) -> List[int]:
    relevant_doc_ids = set()

    for term in query_tokens:
        if term in inverted_index:
            relevant_doc_ids.update(inverted_index[term])

    return list(relevant_doc_ids)


def fetch_tokenized_docs(file_path: str) -> Dict[int, List[str]]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tokenized_docs = json.load(f)

        # Convert the list of lists to a dictionary with ID as key
        doc_dict = {doc["id"]: doc["tokens"] for doc in tokenized_docs}
        return doc_dict
    except Exception as e:
        logger.error(f"Error fetching tokenized documents: {e}")
        return {}


def calculate_query_tf_idf(
    query_tokens: List[str],
    tf_idf_values: Dict[int, Dict[str, float]],
    relevant_doc_ids: List[int],
) -> Dict[int, float]:
    query_tf_idf = defaultdict(float)

    # Calculate the TF-IDF score for each document in the relevant_doc_ids
    for doc_id in relevant_doc_ids:
        doc_tf_idf = tf_idf_values.get(doc_id, {})

        # Calculate the score for the query
        score = sum(
            doc_tf_idf.get(token, 0) * (query_tokens.count(token) / len(query_tokens))
            for token in query_tokens
        )

        if score > 0:
            query_tf_idf[doc_id] = score

    return query_tf_idf


def handle_query(query: str) -> dict[int, List[str]]:
    query_tokens = tokenize_query(query)
    logger.info(f"Tokenized query: {query_tokens}")

    # Fetch the inverted index and TF-IDF values
    inverted_index = fetch_inverted_index(inverted_index_file)
    tf_idf_values = fetch_tf_idf_values(tf_idf_file)

    # Step 1: Retrieve relevant document IDs using the inverted index
    relevant_doc_ids = retrieve_documents_from_index(inverted_index, query_tokens)

    # If there are no relevant documents found, return an empty result
    if not relevant_doc_ids:
        logger.info(f"No relevant documents found for query '{query}'.")
        return []

    # logger.info(f"Initial relevant documents for query '{query}': {relevant_doc_ids}")

    # Step 2: Calculate TF-IDF scores for the query
    query_scores = calculate_query_tf_idf(query_tokens, tf_idf_values, relevant_doc_ids)

    # Step 3: Sort document IDs by their TF-IDF scores and get the top 30
    sorted_docs = sorted(query_scores.items(), key=lambda x: x[1], reverse=True)
    top_doc_ids = [doc_id for doc_id, score in sorted_docs[:100]]

    # Log the top 30 document IDs and their scores
    logger.info(f"Top 30 document IDs for query '{query}': {top_doc_ids}")

    # Fetch tokenized documents
    tokenized_docs = fetch_tokenized_docs(tokenized_docs_file)

    # Retrieve the tokenized content for the top document IDs
    # result = [tokenized_docs.get(doc_id, []) for doc_id in top_doc_ids]
    result = {}
    for key, value in tokenized_docs.items():
        if key in top_doc_ids:
            result[key] = value

    return result


def fetch_all_documents() -> List[Dict[str, str]]:
    try:
        with open("index_documents.json", "r", encoding="utf-8") as file:
            documents = json.load(file)
        return [{"id": doc["id"], "document": doc["document"]} for doc in documents]
    except Exception as e:
        logger.error(f"Error fetching documents: {e}")
        return []
