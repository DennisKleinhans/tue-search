import math
import nltk
import ssl
import re
import json
import sqlite3
import sqlitecloud
from collections import defaultdict
from typing import List, Dict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sql_utils import get_all_documents

# Ensure necessary NLTK resources are downloaded
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Uncomment these lines if running for the first time to download NLTK data
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

def lemmatize_tokens(tokens: List[str]) -> List[str]:
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token.lower()) for token in tokens]

def tokenize_docs(data: List[List[str]]) -> List[List[str]]:
    stop_words = set(stopwords.words('english'))
    tokenized_docs = []
    for doc in data:
        text = doc[3]  # Assuming the text to tokenize is in the 4th column (index 3)
        tokens = word_tokenize(text)
        cleaned_tokens = [token.lower() for token in tokens if re.match(r'^[a-zA-Z0-9äöüß]+$', token)]
        final_tokens = [token for token in cleaned_tokens if token not in stop_words]
        lemmatized_tokens = lemmatize_tokens(final_tokens)
        tokenized_docs.append(lemmatized_tokens)
    return tokenized_docs

def fetch_and_tokenize_documents(api_key: str, db_url: str, db_name: str) -> List[List[str]]:
    try:
        conn = sqlitecloud.connect(f"sqlitecloud://{db_url}?apikey={api_key}")
        conn.execute(f"USE DATABASE {db_name}")
        cursor = conn.cursor()
        documents_from_db = get_all_documents(cursor)
        tokenized_docs_from_db = tokenize_docs(documents_from_db)
        return tokenized_docs_from_db
    except Exception as e:
        print(f"Error: {e}")
        return []
    finally:
        if conn:
            conn.close()

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

def check_table_exists(cursor, table_name: str) -> bool:
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
    return cursor.fetchone() is not None

def create_processed_documents_table(cursor):
    if not check_table_exists(cursor, 'documents_processed'):
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

def store_processed_documents_and_tf_idf(cursor, documents_from_db: List[List[str]], tokenized_docs: List[List[str]], tf_idf_scores: List[Dict[str, float]], idf_scores: Dict[str, float]):
    for doc_id, (doc, tokens, tf_idf) in enumerate(zip(documents_from_db, tokenized_docs, tf_idf_scores)):
        url = doc[0]  # Assuming URL is the first column (index 0)
        processed_text = json.dumps(tokens)  # Convert list of tokens to JSON string
        for token, tf_idf_value in tf_idf.items():
            cursor.execute('''
                INSERT INTO documents_processed (document_id, url, processed_text, token, tf, idf)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (doc_id, url, processed_text, token, tf_idf_value, idf_scores[token]))

# Example usage
api_key = "xZXTNaxWuKM6ryHCVELzSVnT3KC3AubraCDuwFyxKJ4"
db_url = "cyd2d2juiz.sqlite.cloud:8860"
db_name = "documents"

tokenized_docs_from_db = fetch_and_tokenize_documents(api_key, db_url, db_name)

if tokenized_docs_from_db:
    tf_idf_scores = calculate_tf_idf(tokenized_docs_from_db)
    idf_scores = calculate_idf(tokenized_docs_from_db)

    conn = sqlitecloud.connect(f"sqlitecloud://{db_url}?apikey={api_key}")
    conn.execute(f"USE DATABASE {db_name}")
    cursor = conn.cursor()

    #create_processed_documents_table(cursor)
    documents_from_db = get_all_documents(cursor)
    #store_processed_documents_and_tf_idf(cursor, documents_from_db, tokenized_docs_from_db, tf_idf_scores, idf_scores)

    conn.commit()
    conn.close()