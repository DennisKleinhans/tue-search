import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # enable CORS for all routes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query")
    logger.info(f"Received query: {query}")

    # call the preprocessing
    preprocessing_results = preprocessing(query)

    # call the ranking function with the preprocessing and the query as input to get the ranked results
    ranked_results = retriev_ranking(query, preprocessing_results)

    # send the ranked results as a response
    response = {"status": "success", "query": query, "results": ranked_results}
    return jsonify(response)


def preprocessing(query):
    # simulate preprocessing results
    logging.info(f"preprocess index on query: {query}")
    index_results = {
        "documents": [
            {"id": 1, "content": "Document 1 content"},
            {"id": 2, "content": "Document 2 content"},
        ]
    }
    return index_results


def retriev_ranking(query, index_results):
    # simulate ranking results
    print(f"ranking index with: {query} on index results: {index_results}")
    ranked_results = [
        {"id": 1, "content": "Document 1 content", "score": 0.95},
        {"id": 2, "content": "Document 2 content", "score": 0.85},
    ]
    return ranked_results


if __name__ == "__main__":
    app.run(debug=True, port=5000)
