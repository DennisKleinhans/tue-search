#Test to see if all the words are that we retreive from the documents really are in the unique word list we get from the inverted_index
from collections import defaultdict
from typing import Dict, List, Tuple
#Problems with importing preprocessing and inverted_index. If it doesnt work try adding src.indexing before preprocessing and inverted_index. 
from preprocessing import fetch_and_tokenize_documents
from inverted_index import build_inverted_index


api_key = "xZXTNaxWuKM6ryHCVELzSVnT3KC3AubraCDuwFyxKJ4"
db_url = "cyd2d2juiz.sqlite.cloud:8860"
db_name = "documents"

def build_inverted_index(tokenized_docs: List[List[str]]) -> Tuple[Dict[str, List[int]], List[str]]:
    inverted_index = defaultdict(list)
    unique_words = set()  
    
    for doc_id, tokens in enumerate(tokenized_docs):
        for token in tokens:
            unique_words.add(token)  
            inverted_index[token].append(doc_id)
    
    return dict(inverted_index), list(unique_words)

def test_unique_words_retrieval():
    tokenized_docs = fetch_and_tokenize_documents(api_key, db_url, db_name)
    inverted_index, unique_words = build_inverted_index(tokenized_docs)
    
    all_words_from_docs = set()
    for doc_tokens in tokenized_docs:
        all_words_from_docs.update(doc_tokens)
    
    missing_words = all_words_from_docs - set(unique_words)
    
    if missing_words:
        print(f"Missing words in unique_words list: {missing_words}")
    else:
        print("All words from documents are in the unique_words list.")

if __name__ == "__main__":
    test_unique_words_retrieval()