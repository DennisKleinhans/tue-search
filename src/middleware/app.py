import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from src.indexing.inverted_index import handle_query

app = Flask(__name__)
CORS(app)  # enable CORS for all routes

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

    # call the ranking function with the preprocessing and the query as input to get the ranked results
    ranked_results = retrieve_ranking(query, inverted_index_results)

    # send the ranked results as a response
    response = {"status": "success", "query": query, "results": ranked_results}
    return jsonify(response)


def process_batch_queries(queries):
    batch_results = []
    for query_obj in queries:
        query = query_obj["query"]
        query_number = query_obj["queryNumber"]
        logger.info(f"Processing batch query: {query} (queryNumber: {query_number}")

        # process each query individually
        preprocessing_results = preprocessing(query)
        ranked_results = retrieve_ranking(query, preprocessing_results)

        batch_results.append({"queryNumber": query_number, "results": ranked_results})

    # create a response with all batch results
    return jsonify({"status": "success", "batch_results": batch_results})


def preprocessing(query):
    # simulate preprocessing results
    logging.info(f"Preprocessing index on query: {query}")
    index_results = {
        "documents": [
            {"id": 1, "content": "Document 1 content"},
            {"id": 2, "content": "Document 2 content"},
        ]
    }
    return index_results


def retrieve_ranking(query, index_results):
    # simulate ranking results
    logging.info(f"Ranking index with: {query} on index results: {index_results}")
    ranked_results = [
        {"id": 1, "content": "Document 1 content", "score": 0.95},
        {"id": 2, "content": "Document 2 content", "score": 0.85},
    ]
    return ranked_results


if __name__ == "__main__":
    app.run(debug=True, port=5000)
