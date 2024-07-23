import json
import math
import os
import nltk
from nltk.corpus import wordnet
from collections import defaultdict
from typing import Dict, List
from indexing.preprocessing import tokenize_query
import logging

nltk.download("wordnet")
nltk.download("omw-1.4")

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


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace("_", " "))
    return list(synonyms)


def expand_query(query_tokens: List[str]) -> List[str]:
    expanded_tokens = set(query_tokens)
    for token in query_tokens:
        synonyms = get_synonyms(token)
        expanded_tokens.update(synonyms)
    return list(expanded_tokens)


def calculate_document_statistics(tokenized_docs: Dict[int, List[str]]):
    doc_lengths = {}
    total_length = 0

    for doc_id, tokens in tokenized_docs.items():
        length = len(tokens)
        doc_lengths[doc_id] = length
        total_length += length

    avg_doc_length = total_length / len(tokenized_docs)
    return doc_lengths, avg_doc_length


def bm25(term, doc_id, doc_lengths, avg_doc_length, inverted_index, k1=1.5, b=0.75):
    N = len(doc_lengths)  # Total number of documents
    df = (
        len(inverted_index[term]) if term in inverted_index else 0
    )  # Document frequency of term
    idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

    tf = 0
    if term in inverted_index and doc_id in inverted_index[term]:
        tf = inverted_index[term].count(doc_id)  # Term frequency in the document

    doc_length = doc_lengths.get(doc_id, 0)

    score = idf * (
        (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avg_doc_length)))
    )
    return score


def handle_query(query: str) -> dict[int, List[str]]:
    query_tokens = tokenize_query(query)
    logger.info(f"Tokenized query: {query_tokens}")

    # Query Expansion using WordNet
    expanded_query_tokens = expand_query(query_tokens)
    logger.info(f"Expanded query tokens: {expanded_query_tokens}")

    # Fetch the inverted index and TF-IDF values
    inverted_index = fetch_inverted_index(inverted_index_file)
    tf_idf_values = fetch_tf_idf_values(tf_idf_file)

    # Step 1: Retrieve relevant document IDs using the inverted index
    relevant_doc_ids = retrieve_documents_from_index(inverted_index, query_tokens)

    # If there are no relevant documents found, return an empty result
    if not relevant_doc_ids:
        logger.info(f"No relevant documents found for query '{query}'.")
        return []

    # Fetch tokenized documents
    tokenized_docs = fetch_tokenized_docs(tokenized_docs_file)
    doc_lengths, avg_doc_length = calculate_document_statistics(tokenized_docs)

    # Step 2: Calculate TF-IDF scores for the query
    query_scores = calculate_query_tf_idf(query_tokens, tf_idf_values, relevant_doc_ids)

    # Calculate BM25 scores for the query
    bm25_scores = defaultdict(float)
    for term in query_tokens:
        for doc_id in relevant_doc_ids:
            bm25_scores[doc_id] += bm25(
                term, doc_id, doc_lengths, avg_doc_length, inverted_index
            )

    # Combine TF-IDF and BM25 scores
    combined_scores = {
        doc_id: (query_scores.get(doc_id, 0) + bm25_scores.get(doc_id, 0))
        for doc_id in relevant_doc_ids
    }

    # Step 3: Sort document IDs by their TF-IDF scores and get the top 100
    sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    top_doc_ids = [doc_id for doc_id, score in sorted_docs[:100]]

    # Log the top 100 document IDs and their scores
    logger.info(f"Top 100 document IDs for query '{query}': {top_doc_ids}")

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
