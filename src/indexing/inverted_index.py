from preprocessing import fetch_and_tokenize_documents
from collections import defaultdict
from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import Dict, List
from preprocessing import tokenize_query
import sqlitecloud
import logging


app = Flask(__name__)
CORS(app) 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_key = "xZXTNaxWuKM6ryHCVELzSVnT3KC3AubraCDuwFyxKJ4"
db_url = "cyd2d2juiz.sqlite.cloud:8860"
db_name = "documents"

def check_table_exists(cursor, table_name: str) -> bool:
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
    return cursor.fetchone() is not None

def create_inverted_index_table(cursor):
    if not check_table_exists(cursor, 'inverted_index_list'):
        cursor.execute('''
            CREATE TABLE inverted_index_list (
                term TEXT PRIMARY KEY,
                doc_ids TEXT
            );
        ''')
        print("Table inverted_index_list created.")
    else:
        print("Table inverted_index_list already exists.")

def build_inverted_index(tokenized_docs: List[List[str]]) -> Dict[str, List[int]]:
    inverted_index = defaultdict(list)
    
    for doc_id, tokens in enumerate(tokenized_docs):
        for token in set(tokens):  
            inverted_index[token].append(doc_id)
    
    return inverted_index

def store_inverted_index(cursor, inverted_index: Dict[str, List[int]]):
    for term, doc_ids in inverted_index.items():
        doc_ids_str = ','.join(map(str, doc_ids))
        cursor.execute('''
            INSERT OR REPLACE INTO inverted_index_list (term, doc_ids)
            VALUES (?, ?)
        ''', (term, doc_ids_str))

def run_inverted_index_process():

    tokenized_docs_from_db = fetch_and_tokenize_documents(api_key, db_url, db_name)
    
    if tokenized_docs_from_db:
        inverted_index = build_inverted_index(tokenized_docs_from_db)
        try:
            conn = sqlitecloud.connect(f"sqlitecloud://{db_url}?apikey={api_key}")
            conn.execute(f"USE DATABASE {db_name}")
            cursor = conn.cursor()
            
            create_inverted_index_table(cursor)
            store_inverted_index(cursor, inverted_index)
            
            conn.commit()
            print("Inverted index stored successfully.")
        
        except Exception as e:
            print(f"Error: {e}")
        
        finally:
            if conn:
                conn.close()

run_inverted_index_process()

#Test if the database is being filled properly with the results (comment this code after test)
def fetch_inverted_index(api_key: str, db_url: str, db_name: str) -> Dict[str, List[int]]:
    try:
        conn = sqlitecloud.connect(f"sqlitecloud://{db_url}?apikey={api_key}")
        conn.execute(f"USE DATABASE {db_name}")
        cursor = conn.cursor()
        
        cursor.execute('SELECT term, doc_ids FROM inverted_index_list')
        inverted_index = {}
        for row in cursor.fetchall():
            term = row[0]
            doc_ids_str = row[1]
            doc_ids = list(map(int, doc_ids_str.split(',')))
            inverted_index[term] = doc_ids
        
        return inverted_index
    
    except Exception as e:
        print(f"Error: {e}")
        return {}
    
    finally:
        if conn:
            conn.close()

inverted_index = fetch_inverted_index(api_key, db_url, db_name)
print(inverted_index)

def retrieve_documents_from_index(inverted_index: Dict[str, List[int]], query_tokens: List[str]) -> List[int]:
    relevant_doc_ids = set()

    for term in query_tokens:
        if term in inverted_index:
            relevant_doc_ids.update(inverted_index[term])
    
    return list(relevant_doc_ids)

#TODO receive the query preprocess it and compare to the inverted_index list of unique words.
def handle_query(query):
    query_tokens = tokenize_query(query)
    logger.info(f"Preprocessed query tokens: {query_tokens}")

    relevant_doc_ids = retrieve_documents_from_index(inverted_index, query_tokens)
    logger.info(f"Relevant documents for query '{query}': {relevant_doc_ids}")

    return relevant_doc_ids

#TODO return the results to retrieval_system as tokenized docs.