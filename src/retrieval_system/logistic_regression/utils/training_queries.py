import random
from datasets import Dataset
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# Loading the Spacy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print(
        "The Spacy model 'en_core_web_sm' must first be downloaded. Execute 'python -m spacy download en_core_web_sm'."
    )
    exit()


def extract_named_entities(tokens: list) -> list:
    """
    Extract named entities from a list of tokens using Spacy.
    """
    doc = nlp(" ".join(tokens))
    entities = [ent.text for ent in doc.ents]
    return entities


def extract_keywords(documents: dict) -> dict:
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


def generate_n_grams(tokens: list, n: int) -> list:
    """
    Generate n-grams from the given list of tokens.
    """
    n_grams = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return n_grams


def document_contains_query(doc_tokens, query_tokens):
    return all(token in doc_tokens for token in query_tokens)


def generate_training_data_batch(
    documents: dict,
    num_queries: int = 5,
    negative_ratio: float = 0.5,
    num_documents_per_query: int = 5,
) -> Dataset:
    """
    Generate training data for a ranking model.

    Parameters:
    - documents: dict, a dictionary with document ids as keys and tokenized document contents as values
    - num_queries: int, number of queries to generate per document
    - negative_ratio: float, ratio of negative examples to generate (e.g., 0.5 means half of the queries are negative)

    Returns:
    - A Dataset object with fields 'query', 'document', and 'flag'
    """
    data = {"query": [], "document": [], "lable": []}

    doc_ids = list(documents.keys())
    keywords = extract_keywords(documents)

    for _ in range(num_queries):
        # Wähle zufällig ein Dokument aus und extrahiere daraus eine Query
        doc_id = random.choice(doc_ids)
        tokens = documents[doc_id]

        # Extrahiere benannte Entitäten, n-Grams und Schlüsselwörter für die Query
        named_entities = extract_named_entities(tokens)
        n_grams = generate_n_grams(tokens, random.randint(3, 5))
        possible_queries = named_entities + n_grams + keywords[doc_id]

        # Filtere nach mindestens 3 Tokens
        possible_queries = [
            query for query in possible_queries if len(query.split()) >= 3
        ]

        if not possible_queries:
            continue  # Falls keine gültigen Queries gefunden werden, überspringen

        query = random.choice(possible_queries)
        query_tokens = query.split()
        data["query"].append(query)

        # Wähle positive Dokumente aus, die die Query-Tokens enthalten
        positive_doc_ids = [
            doc_id
            for doc_id in doc_ids
            if document_contains_query(documents[doc_id], query_tokens)
        ]

        # Falls nicht genügend positive Dokumente gefunden werden, überspringen
        if len(positive_doc_ids) < int(num_documents_per_query * (1 - negative_ratio)):
            continue

        num_positive_documents = min(
            int(num_documents_per_query * (1 - negative_ratio)), len(positive_doc_ids)
        )
        num_negative_documents = num_documents_per_query - num_positive_documents

        positive_doc_ids = random.sample(positive_doc_ids, num_positive_documents)
        negative_doc_ids = random.sample(
            [d for d in doc_ids if d not in positive_doc_ids], num_negative_documents
        )

        documents_list = [
            " ".join(documents[doc_id])
            for doc_id in positive_doc_ids + negative_doc_ids
        ]
        flags_list = [1] * len(positive_doc_ids) + [0] * len(negative_doc_ids)

        data["document"].append(documents_list)
        data["lable"].append(flags_list)

    return data


def generate_training_data(
    documents,
    batch_size=100,
    num_queries=5,
    num_documents_per_query=3,
    negative_ratio=0.5,
) -> Dataset:
    """
    Generate training data for a logistic regression model.

    This function takes a dictionary of documents as input and generates training data
    for a logistic regression model. It divides the documents into batches of size `batch_size`
    and generates training data for each batch using the `generate_training_data_batch` function.
    The training data is then concatenated to form a full dataset.

    Parameters:
    - documents (dict): A dictionary of documents where the keys are document IDs and the values
        are the corresponding document contents.
    - batch_size (int, optional): The size of each batch. Defaults to 100.
    - num_queries (int, optional): The number of queries to generate for each batch. Defaults to 5.
    - num_documents_per_query (int, optional): The number of documents to include per query. Defaults to 3.
    - negative_ratio (float, optional): The ratio of negative samples to include in the training data.
        Negative samples are documents that are not relevant to the query. Defaults to 0.5.

    Returns:
    - full_dataset (Dataset): The full training dataset containing the generated training data for all batches.
    """
    total_docs = len(documents)
    doc_ids = list(documents.keys())
    batches = [doc_ids[i : i + batch_size] for i in range(0, total_docs, batch_size)]

    all_queries = []
    all_documents = []
    all_flags = []

    for batch in batches:
        batch_documents = {doc_id: documents[doc_id] for doc_id in batch}
        batch_data = generate_training_data_batch(
            batch_documents, num_queries, num_documents_per_query, negative_ratio
        )
        all_queries.extend(batch_data["query"])
        all_documents.extend(batch_data["document"])
        all_flags.extend(batch_data["lable"])

    dataset = Dataset.from_dict(
        {"query": all_queries, "document": all_documents, "lable": all_flags}
    )
    return dataset
