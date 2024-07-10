from preprocessing import fetch_and_tokenize_documents
from collections import defaultdict
from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import Dict, List
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

#Test if the database is being filled properly with the results
'''def fetch_inverted_index(api_key: str, db_url: str, db_name: str) -> Dict[str, List[int]]:
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
            conn.close()'''

#inverted_index = fetch_inverted_index(api_key, db_url, db_name)
#print(inverted_index)


#TODO Potentially move this code to the app.py 
'''@app.route("/search", methods=["POST"])
def search():
    global inverted_index
    
    data = request.get_json()
    if "query" in data:
        query = data.get("query")
        logger.info(f"Received query: {query}")
        
        relevant_doc_ids = retrieve_documents_from_index(inverted_index, query)

        # For demonstration, returning just document IDs
        return jsonify({"status": "success", "query": query, "results": relevant_doc_ids})
    else:
        return jsonify({"error": "Missing 'query' parameter"}), 400

@app.errorhandler(404)
def page_not_found(error):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({"error": "Internal server error"}), 500

def retrieve_documents_from_index(inverted_index: Dict[str, List[int]], query: str) -> List[int]:
    query_terms = query.lower().split()
    relevant_doc_ids = set()

    for term in query_terms:
        if term in inverted_index:
            relevant_doc_ids.update(inverted_index[term])
    
    return list(relevant_doc_ids)

if __name__ == "__main__":
    app.run(debug=True, port=5000)'''

#TODO receive the query preprocess it and compare to the inverted_index list of unique words.
#TODO return the results to retrieval_system.