from collections import defaultdict
from typing import List, Dict
import sqlitecloud
import math
import json

#This can be used when the project is set up to increase Quality of query results

#TODO Check calculations again
def calculate_tf(doc_tokens: List[str]) -> Dict[str, float]:
    tf_scores = defaultdict(float)
    total_tokens = len(doc_tokens)
    for token in doc_tokens:
        tf_scores[token] += 1
    for token in tf_scores:
        tf_scores[token] /= total_tokens
    return tf_scores

#TODO Check calculations again
def calculate_idf(tokenized_docs: List[List[str]]) -> Dict[str, float]:
    idf_scores = defaultdict(float)
    total_docs = len(tokenized_docs)
    for doc_tokens in tokenized_docs:
        unique_tokens = set(doc_tokens)
        for token in unique_tokens:
            idf_scores[token] += 1
    for token in idf_scores:
        idf_scores[token] = math.log(total_docs / (1 + idf_scores[token]))
    return idf_scores

#TODO Check calculations again
def calculate_tf_idf(tokenized_docs: List[List[str]]) -> List[Dict[str, float]]:
    idf_scores = calculate_idf(tokenized_docs)
    tf_idf_scores = []
    for doc_tokens in tokenized_docs:
        tf_scores = calculate_tf(doc_tokens)
        tf_idf = {token: tf_scores[token] * idf_scores[token] for token in tf_scores}
        tf_idf_scores.append(tf_idf)
    return tf_idf_scores

#tf_idf_scores = calculate_tf_idf(tokenized_docs_from_db)
#idf_scores = calculate_idf(tokenized_docs_from_db)

def check_table_exists(cursor, table_name: str) -> bool:
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
    return cursor.fetchone() is not None

def create_tfidf_documents_table(cursor):
    if not check_table_exists(cursor, 'documents_tfidf'):
        cursor.execute('''
            CREATE TABLE documents_processed (
                document_id INTEGER PRIMARY KEY,
                url TEXT,
                processed_text TEXT,
                token TEXT,
                tf REAL,
                idf REAL
            );
        ''')
        print("Table documents_processed created.")
    else:
        print("Table documents_processed already exists.")

def store_tf_idf(cursor, documents_from_db: List[List[str]], tokenized_docs: List[List[str]], tf_idf_scores: List[Dict[str, float]], idf_scores: Dict[str, float]):
    for doc_id, (doc, tokens, tf_idf) in enumerate(zip(documents_from_db, tokenized_docs, tf_idf_scores)):
        url = doc[0]
        processed_text = json.dumps(tokens)  # Convert list of tokens to JSON string
        for token, tf_idf_value in tf_idf.items():
            cursor.execute('''
                INSERT INTO documents_processed (document_id, url, processed_text, token, tf, idf)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (doc_id, url, processed_text, token, tf_idf_value, idf_scores[token]))
            

'''def add_tf_idf_values():
    db_name = "documents"

    conn = sqlitecloud.connect(f"sqlitecloud://{db_url}?apikey={api_key}")
    conn.execute(f"USE DATABASE {db_name}")
    cursor = conn.cursor()
    #create_tfidf_documents_table(cursor)
    #store_tf_idf(cursor, documents_from_db, tokenized_docs_from_db, tf_idf_scores, idf_scores)
    conn.commit()
    conn.close()'''
