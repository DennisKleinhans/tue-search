from src.indexing.build_inverted_index import run_inverted_index_process
from src.indexing.preprocessing import build_tokenized_docs
from src.indexing.build_tf_and_idf import main as build_tf_and_idf
from src.indexing.build_tf_idf import main as build_tf_idf


def build_complete_index():
    build_tokenized_docs()
    build_tf_and_idf()
    build_tf_idf()
    run_inverted_index_process()


if __name__ == "__main__":
    build_complete_index()
