import json
import os
from typing import Dict, List

# Interpreting TF-IDF Values:
# 
# TF-IDF (Term Frequency-Inverse Document Frequency) is a measure used to evaluate the importance of a term in a document relative to a corpus.
# 
# - **Term Frequency (TF)**: Indicates how often a term appears in a document.
#   - Higher TF means the term appears more frequently in the document.
# 
# - **Inverse Document Frequency (IDF)**: Indicates how rare or common a term is across the entire corpus.
#   - Higher IDF means the term is rare across documents.
# 
# - **TF-IDF**: Combines TF and IDF to give a composite score.
#   - Higher TF-IDF indicates that the term is frequent in the specific document but rare across the corpus, suggesting high importance.
#   - Lower TF-IDF indicates that the term is either infrequent in the document or common across the corpus, suggesting lower importance.
# 
# Practical Usage:
# - **Document Relevance**: Rank documents based on TF-IDF scores for query terms; higher scores indicate higher relevance.
# - **Keyword Extraction**: Identify key terms in documents by looking at terms with high TF-IDF scores.
# - **Document Clustering**: Group similar documents by comparing their TF-IDF vectors; similar documents will have similar TF-IDF values.

# File paths
results_folder = "results"
tf_file = os.path.join(results_folder, "tf_values.json")
idf_file = os.path.join(results_folder, "idf_values.json")
tf_idf_file = os.path.join(results_folder, "tf_idf_values.json")

def load_json(file_path: str) -> Dict:
   with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def calculate_tf_idf(tf_values: Dict[int, Dict[str, float]], idf_values: Dict[str, float]) -> Dict[int, Dict[str, float]]:
    tf_idf_scores = {}
    for doc_id, tf_scores in tf_values.items():
        tf_idf = {token: tf_scores[token] * idf_values.get(token, 0) for token in tf_scores}
        tf_idf_scores[doc_id] = tf_idf
    return tf_idf_scores

def store_json(data: Dict, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def main():
    # Load TF and IDF values from JSON files
    tf_values = load_json(tf_file)
    idf_values = load_json(idf_file)

    # Calculate TF-IDF scores
    tf_idf_values = calculate_tf_idf(tf_values, idf_values)

    # Store TF-IDF scores in a JSON file
    store_json(tf_idf_values, tf_idf_file)

    print(f"TF-IDF values stored in {tf_idf_file}")

if __name__ == "__main__":
    main()