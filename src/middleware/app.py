import json
import logging
import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.append(os.path.abspath("src"))
from indexing.process_query import handle_query
from retrieval_system.logistic_regression.retrieval_interface import (
    RetrievalSystemInterface,
)


app = Flask(__name__)
CORS(app)  # enable CORS for all routes

# init RSI
RSI = RetrievalSystemInterface()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    if "query" in data:
        query = data.get("query")
        logger.info(f"Received query: {query}")
        return process_single_query(query)
    else:
        return jsonify({"error": "Missing 'query' parameter"}), 400


@app.route("/batch_search", methods=["POST"])
def batch_search():
    data = request.get_json()
    if "queries" in data:
        queries = data.get("queries")
        logger.info(f"Received batch queries: {queries}")
        return process_batch_queries(queries)
    else:
        return jsonify({"error": "Missing 'queries' parameter"}), 400


def process_single_query(query):
    # call the preprocessing
    inverted_index_results = handle_query(query)

    # Extract the tokenized documents and their corresponding IDs
    doc_ids = list(inverted_index_results.keys())
    tokenized_docs = list(inverted_index_results.values())

    # Call the ranking function with the tokenized documents, query, and IDs to get the ranked results
    ranked_results_with_ids = RSI.retrieve_ranking(query, tokenized_docs, doc_ids)
    # Load the JSON file containing the documents
    with open("index_documents.json", "r") as file:
        documents = json.load(file)

    # Create a mapping from ID to URL
    id_to_url = {doc["id"]: doc["url"] for doc in documents}

    # Build the response with URLs
    results_with_urls = []
    for document, score, doc_id in ranked_results_with_ids:
        url = id_to_url.get(doc_id, None)
        if url:
            results_with_urls.append({"id": doc_id, "score": score, "url": url})

    # Build the final response
    response = {"status": "success", "query": query, "results": results_with_urls}
    return jsonify(response)


def process_batch_queries(queries):
    batch_results = []
    for query_obj in queries:
        query = query_obj["query"]
        query_number = query_obj["queryNumber"]
        logger.info(f"Processing batch query: {query} (queryNumber: {query_number}")

        # process each query individually
        preprocessing_results = handle_query(query)

        ranked_results = RSI.retrieve_ranking(query, preprocessing_results)

        batch_results.append({"queryNumber": query_number, "results": ranked_results})

    # create a response with all batch results
    return jsonify({"status": "success", "batch_results": batch_results})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
