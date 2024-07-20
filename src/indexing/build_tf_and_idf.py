from collections import defaultdict
from typing import List, Dict
import math
import json
import os


results_folder = "results"
inverted_index_file = os.path.join(results_folder, "inverted_index.json")
tokenized_docs_file = os.path.join(results_folder, "tokenized_docs.json")
tf_file = os.path.join(results_folder, "tf_values.json")
idf_file = os.path.join(results_folder, "idf_values.json")

def load_tokenized_docs(file_path: str) -> List[List[str]]:
    with open(file_path, 'r', encoding='utf-8') as file:
        tokenized_docs = json.load(file)
    return tokenized_docs


def load_inverted_index(file_path: str) -> Dict[str, List[int]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            inverted_index = json.load(file)
        
        return inverted_index
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}


def store_json(data: Dict, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def calculate_tf(doc_tokens: List[str]) -> Dict[str, float]:
    tf_scores = defaultdict(float)
    total_tokens = len(doc_tokens)
    for token in doc_tokens:
        tf_scores[token] += 1
    for token in tf_scores:
        tf_scores[token] /= total_tokens
    return tf_scores


def calculate_idf(tokenized_docs: List[Dict[str, List[str]]]) -> Dict[str, float]:
    idf_scores = defaultdict(float)
    total_docs = len(tokenized_docs)
    for doc in tokenized_docs:
        unique_tokens = set(doc["tokens"])
        for token in unique_tokens:
            idf_scores[token] += 1
    for token in idf_scores:
        idf_scores[token] = math.log(total_docs / (1 + idf_scores[token]))
    return idf_scores

def main():
    # Load tokenized documents
    tokenized_docs = load_tokenized_docs(tokenized_docs_file)

    # Calculate TF values for each document
    tf_values = {}
    for doc in tokenized_docs:
        doc_id = doc["id"]
        tf_values[doc_id] = calculate_tf(doc["tokens"])

    # Store TF values in a JSON file
    store_json(tf_values, tf_file)

    # Calculate IDF values for the entire corpus
    idf_values = calculate_idf(tokenized_docs)

    # Store IDF values in a JSON file
    store_json(idf_values, idf_file)

    print(f"TF values stored in {tf_file}")
    print(f"IDF values stored in {idf_file}")

if __name__ == "__main__":
    main()