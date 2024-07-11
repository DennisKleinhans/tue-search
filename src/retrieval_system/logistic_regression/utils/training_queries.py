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


def generate_training_data(
    documents: dict, num_queries: int = 5, negative_ratio: float = 0.5
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
    data = {"query": [], "document": [], "flag": []}

    doc_ids = list(documents.keys())
    keywords = extract_keywords(documents)

    for doc_id, tokens in documents.items():
        num_positive_queries = int(num_queries * (1 - negative_ratio))
        num_negative_queries = num_queries - num_positive_queries

        # Extract named entities and n-grams
        named_entities = extract_named_entities(tokens)
        n_grams = generate_n_grams(tokens, random.randint(3, 5))

        # Combination of named entities and n-grams for positive queries
        possible_queries = named_entities + n_grams + keywords[doc_id]

        # Filter by at least 2 tokens
        possible_queries = [
            query for query in possible_queries if len(query.split()) >= 2
        ]

        # Generation of positive queries
        positive_queries = random.sample(
            possible_queries, min(num_positive_queries, len(possible_queries))
        )
        for query in positive_queries:
            data["query"].append(query)
            data["document"].append(" ".join(tokens))
            data["flag"].append(1)

        # Generation of negative queries
        for _ in range(num_negative_queries):
            negative_doc_id = random.choice([d for d in doc_ids if d != doc_id])
            negative_tokens = documents[negative_doc_id]
            negative_named_entities = extract_named_entities(negative_tokens)
            negative_n_grams = generate_n_grams(negative_tokens, random.randint(3, 5))
            negative_possible_queries = (
                negative_named_entities + negative_n_grams + keywords[negative_doc_id]
            )

            # Filter by at least 2 tokens
            negative_possible_queries = [
                query for query in negative_possible_queries if len(query.split()) >= 2
            ]

            if negative_possible_queries:
                negative_query = random.choice(negative_possible_queries)
                data["query"].append(negative_query)
                data["document"].append(" ".join(tokens))
                data["flag"].append(0)

    dataset = Dataset.from_dict(data)
    return dataset
