import nltk
import os
import ssl
import re
from typing import List, Dict, Tuple
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from crawler.sql_utils import get_all_documents, get_first_10_documents
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

lemmatizer = WordNetLemmatizer()

def lemmatize_tokens(tokens: List[str]) -> List[str]:
    return [lemmatizer.lemmatize(token.lower()) for token in tokens]

def tokenize_docs(data: List[Tuple[int, str]]) -> List[Dict[str, List[str]]]:
    stop_words = set(stopwords.words("english"))
    tokenized_docs = []
    logger.info("Started tokenizing docs")
    for doc_id, text in data:
        tokens = word_tokenize(text)
        cleaned_tokens = [token.lower() for token in tokens if re.match(r'^[a-zA-Zäöüß]+$', token) and token.lower() not in stop_words]
        lemmatized_tokens = lemmatize_tokens(cleaned_tokens)
        tokenized_docs.append({"id": doc_id, "tokens": lemmatized_tokens})
    logger.info("Finished tokenizing docs")
    return tokenized_docs

def tokenize_query(text: str) -> List[str]:
    logger.info("Started tokenizing Query")
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    cleaned_tokens = [token.lower() for token in tokens if re.match(r'^[a-zA-Zäöüß]+$', token) and token.lower() not in stop_words]
    lemmatized_tokens = lemmatize_tokens(cleaned_tokens)
    return lemmatized_tokens

def fetch_and_tokenize_documents() -> List[Dict[str, List[str]]]:
    try:
        with open('index_documents.json', 'r', encoding='utf-8') as file:
            documents = json.load(file)
        
        # Check if documents have 'id', otherwise use index as ID
        if 'id' in documents[0]:
            data = [(doc["id"], doc["document"]) for doc in documents]
        else:
            data = [(index, doc["document"]) for index, doc in enumerate(documents)]
        
        tokenized_docs = tokenize_docs(data)
        return tokenized_docs
    except Exception as e:
        logger.error(f"Error fetching and tokenizing documents: {e}")
        return []


# Uncomment the code below to write tokenized documents to a JSON file
#Function is only used if the code below the function is uncommented
def write_tokenized_docs_to_json(tokenized_docs: List[Dict[str, List[str]]], file_path: str) -> None:
    """
    Write tokenized documents to a JSON file.

    Parameters:
    tokenized_docs (List[Dict[str, List[str]]]): List of tokenized documents with IDs
    file_path (str): Path to the output JSON file
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(tokenized_docs, file, ensure_ascii=False, indent=4)
        logger.info(f"Successfully wrote tokenized documents to {file_path}")
    except Exception as e:
        logger.error(f"Error writing tokenized documents to JSON file: {e}")

# Check tokenized words in tokenized_docs.json
#results_folder = "results"
#file = os.path.join(results_folder, "tokenized_docs.json")
#docs = fetch_and_tokenize_documents()
#write_tokenized_docs_to_json(docs, file)