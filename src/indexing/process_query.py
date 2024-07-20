import json
import os
from collections import defaultdict
from typing import Dict, List
from indexing.preprocessing import fetch_and_tokenize_documents, tokenize_query
import logging

logging.basicConfig(level=logging.INFO)  # Use DEBUG level for detailed logs
logger = logging.getLogger(__name__)

results_folder = "results"
inverted_index_file = os.path.join(results_folder, "inverted_index.json")
tokenized_docs_file = os.path.join(results_folder, "tokenized_docs.json")


def fetch_inverted_index(file_path: str) -> Dict[str, List[int]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            inverted_index = json.load(f)
        # Convert lists from strings to integers
        inverted_index = {k: list(map(int, v)) for k, v in inverted_index.items()}
        return inverted_index
    except Exception as e:
        logger.error(f"Error fetching inverted index: {e}")
        return {}


def retrieve_documents_from_index(inverted_index: Dict[str, List[int]], query_tokens: List[str]) -> List[int]:
    relevant_doc_ids = set()
    
    for term in query_tokens:
        if term in inverted_index:
            relevant_doc_ids.update(inverted_index[term])
    
    logger.debug(f"Retrieved document IDs for query tokens '{query_tokens}': {relevant_doc_ids}")
    return list(relevant_doc_ids)


def fetch_tokenized_docs(file_path: str) -> Dict[int, List[str]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tokenized_docs = json.load(f)
        
        # Convert the list of lists to a dictionary with ID as key
        doc_dict = {doc["id"]: doc["tokens"] for doc in tokenized_docs}
        return doc_dict
    except Exception as e:
        logger.error(f"Error fetching tokenized documents: {e}")
        return {}


def handle_query(query: str) -> List[List[str]]:
    query_tokens = tokenize_query(query)

    inverted_index = fetch_inverted_index(inverted_index_file)
    relevant_doc_ids = retrieve_documents_from_index(inverted_index, query_tokens)
    #logger.info(f"Relevant documents for query '{query}': {relevant_doc_ids}")
    
    # Fill with more documents if fewer than 10 are found
    if len(relevant_doc_ids) < 10:
        all_docs = fetch_all_documents()
        additional_ids = [doc["id"] for doc in all_docs if doc["id"] not in relevant_doc_ids]
        relevant_doc_ids.extend(additional_ids[:10 - len(relevant_doc_ids)])
    
    # Limit the number of document IDs to 200
    relevant_doc_ids = relevant_doc_ids[:200]
    
    # Fetch tokenized documents
    tokenized_docs = fetch_tokenized_docs(tokenized_docs_file)
    
    # Retrieve the tokenized content for the relevant document IDs
    result = [tokenized_docs.get(doc_id, []) for doc_id in relevant_doc_ids]
    
    return result


def fetch_all_documents() -> List[Dict[str, str]]:
    try:
        with open('index_documents.json', 'r', encoding='utf-8') as file:
            documents = json.load(file)
        return [{"id": doc["id"], "document": doc["document"]} for doc in documents]
    except Exception as e:
        logger.error(f"Error fetching documents: {e}")
        return []