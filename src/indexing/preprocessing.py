import nltk
import ssl
import re
import sqlitecloud
from typing import List, Dict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sql_utils import get_all_documents, get_first_10_documents

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

#TODO query processing needs to be the same process as document preprocessing

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
        #TODO Change this method to get_all_documents when done testing
        documents_from_db = get_first_10_documents(cursor)
        tokenized_docs_from_db = tokenize_docs(documents_from_db)
        return tokenized_docs_from_db
    except Exception as e:
        print(f"Error: {e}")
        return []
    finally:
        if conn:
            conn.close()

'''
api_key = "xZXTNaxWuKM6ryHCVELzSVnT3KC3AubraCDuwFyxKJ4"
db_url = "cyd2d2juiz.sqlite.cloud:8860"
db_name = "documents"

tokenized_docs_from_db = fetch_and_tokenize_documents(api_key, db_url, db_name)

if tokenized_docs_from_db:

    conn = sqlitecloud.connect(f"sqlitecloud://{db_url}?apikey={api_key}")
    conn.execute(f"USE DATABASE {db_name}")
    cursor = conn.cursor()

    #create_processed_documents_table(cursor)
    #store_processed_documents_and_tf_idf(cursor, documents_from_db, tokenized_docs_from_db, tf_idf_scores, idf_scores)

    conn.commit()
    conn.close()
'''

