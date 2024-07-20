import json
import os
from collections import defaultdict
from typing import Dict, List
from indexing.preprocessing import fetch_and_tokenize_documents, tokenize_query
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

results_folder = "results"
inverted_index_file = os.path.join(results_folder, "inverted_index.json")

def build_inverted_index(tokenized_docs: List[List[str]]) -> Dict[str, List[int]]:
    inverted_index = defaultdict(list)
    
    for doc_id, tokens in enumerate(tokenized_docs):
        #logger.debug(f"Processing document ID {doc_id} with tokens {tokens}")
        for token in set(tokens):
            inverted_index[token].append(doc_id)
    
    return inverted_index

def store_inverted_index(inverted_index: Dict[str, List[int]], file_path: str):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as f:
            json.dump(inverted_index, f, ensure_ascii=False, indent=4)
        logger.info(f"Inverted index stored successfully in {file_path}")
    except Exception as e:
        logger.error(f"Error storing inverted index: {e}")

def fetch_inverted_index(file_path: str) -> Dict[str, List[int]]:
    try:
        with open(file_path, 'r') as f:
            inverted_index = json.load(f)
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
    
    return list(relevant_doc_ids)

def run_inverted_index_process():
    tokenized_docs_from_json = fetch_and_tokenize_documents()
    logger.info(f"Fetched {len(tokenized_docs_from_json)} tokenized documents")
    
    if tokenized_docs_from_json:
        inverted_index = build_inverted_index(tokenized_docs_from_json)
        logger.info(f"Built inverted index with {len(inverted_index)} terms")
        store_inverted_index(inverted_index, inverted_index_file)

def handle_query(query):
    query_tokens = tokenize_query(query)
    #logger.info(f"Preprocessed query tokens: {query_tokens}")
    
    inverted_index = fetch_inverted_index(inverted_index_file)
    relevant_doc_ids = retrieve_documents_from_index(inverted_index, query_tokens)
    logger.info(f"Relevant documents for query '{query}': {relevant_doc_ids}")
    
    return relevant_doc_ids

if __name__ == "__main__":
    run_inverted_index_process()
    inverted_index = fetch_inverted_index(inverted_index_file)
    logger.info(f"Loaded inverted index with {len(inverted_index)} terms")
    #print(json.dumps(inverted_index, indent=4))