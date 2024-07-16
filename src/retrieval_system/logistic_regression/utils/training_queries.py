import random
from datasets import Dataset
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import sqlitecloud

# Loading the Spacy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print(
        "The Spacy model 'en_core_web_sm' must first be downloaded. Execute 'python -m spacy download en_core_web_sm'."
    )
    exit()


def __extract_named_entities(tokens: list) -> list:
    """
    Extract named entities from a list of tokens using Spacy.
    """
    doc = nlp(" ".join(tokens))
    entities = [ent.text for ent in doc.ents]
    return entities


def __extract_keywords(documents: dict) -> dict:
    """
    Extract keywords from the documents using TF-IDF.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(
        [" ".join(tokens) for tokens in documents.values()]
    )
    feature_names = vectorizer.get_feature_names_out()
    keywords = {doc_id: [] for doc_id in documents.keys()}

    for doc_id, row in zip(documents.keys(), tfidf_matrix):
        indices = row.nonzero()[1]
        keyword_indices = sorted(indices, key=lambda idx: row[0, idx], reverse=True)[:5]
        keywords[doc_id] = [feature_names[idx] for idx in keyword_indices]

    return keywords


def __generate_n_grams(tokens: list, n: int) -> list:
    """
    Generate n-grams from the given list of tokens.
    """
    n_grams = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return n_grams


def __document_contains_query(doc_tokens, query_tokens):
    return all(token in doc_tokens for token in query_tokens)


def __get_documents_form_db() -> dict:
    api_key = "xZXTNaxWuKM6ryHCVELzSVnT3KC3AubraCDuwFyxKJ4"
    db_url = "cyd2d2juiz.sqlite.cloud:8860"
    db_name = "documents"

    conn = sqlitecloud.connect(f"sqlitecloud://{db_url}?apikey={api_key}")
    conn.execute(f"USE DATABASE {db_name}")
    cursor = conn.cursor()

    cursor.execute(
        """
    SELECT document_id, token
    FROM documents_processed
    """
    )

    rows = cursor.fetchall()
    documents = {row[0]: row[1] for row in rows}

    return documents


def __generate_training_data(
    documents: dict, num_documents_per_query: int = 5, negative_ratio: float = 0.5
) -> Dataset:
    """
    Generate training data for a ranking model.

    Parameters:
    - documents: dict, a dictionary with document ids as keys and tokenized document contents as values
    - num_documents_per_query: int, number of documents (both positive and negative) to generate per query
    - negative_ratio: float, ratio of negative examples to generate (e.g., 0.5 means half of the queries are negative)

    Returns:
    - A Dataset object with fields 'query', 'document', and 'lable'
    """
    data = {"query": [], "document": [], "lable": []}

    doc_ids = list(documents.keys())
    keywords = __extract_keywords(documents)

    # Define the number of queries to generate based on the number of documents
    num_queries = len(documents) // num_documents_per_query

    for _ in range(num_queries):
        # Choose a random document and extract a query from it
        doc_id = random.choice(doc_ids)
        tokens = documents[doc_id]

        # Extract named entities, n-grams, and keywords for the query
        named_entities = __extract_named_entities(tokens)
        n_grams = __generate_n_grams(tokens, random.randint(3, 5))
        possible_queries = named_entities + n_grams + keywords.get(doc_id, [])

        # Filter queries to ensure at least 3 tokens
        possible_queries = [
            query for query in possible_queries if len(query.split()) >= 3
        ]

        if not possible_queries:
            continue  # Skip if no valid queries found

        query = random.choice(possible_queries)
        query_tokens = query.split()
        data["query"].append(query)

        # Choose positive documents that contain the query tokens
        positive_doc_ids = [
            doc_id
            for doc_id in doc_ids
            if __document_contains_query(documents[doc_id], query_tokens)
        ]

        if not positive_doc_ids:
            continue  # Skip if no positive documents found

        # Choose positive documents up to the limit or available documents
        num_positive_documents = min(
            len(positive_doc_ids), int(num_documents_per_query * (1 - negative_ratio))
        )
        positive_doc_ids = random.sample(positive_doc_ids, num_positive_documents)

        # Choose negative documents
        num_negative_documents = num_documents_per_query - num_positive_documents
        negative_doc_ids = random.sample(
            [d for d in doc_ids if d not in positive_doc_ids], num_negative_documents
        )

        documents_list = [
            "".join(documents[doc_id]) for doc_id in positive_doc_ids + negative_doc_ids
        ]
        flags_list = [1] * len(positive_doc_ids) + [0] * len(negative_doc_ids)

        data["document"].append(documents_list)
        data["lable"].append(flags_list)

    dataset = Dataset.from_dict(data)
    return dataset


def get_synthetic_training_data():
    documents = __get_documents_form_db()
    dataset = __generate_training_data(documents=documents, num_documents_per_query=1)
    return dataset


dataset = get_synthetic_training_data()

for i in range(len(dataset)):
    print(f"Query: {dataset[i]['query']}")
    for doc, label in zip(dataset[i]["document"], dataset[i]["lable"]):
        print(f"  Document: {doc}")
        print(f"  Label: {label}")
    print()
