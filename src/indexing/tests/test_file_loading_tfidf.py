from indexing.build_tf_and_idf import load_tokenized_docs, load_inverted_index
import os

results_folder = "results"
inverted_index_file = os.path.join(results_folder, "inverted_index.json")
tokenized_docs_file = os.path.join(results_folder, "tokenized_docs.json")

def test_file_loading():
    try:
        tokenized_docs = load_tokenized_docs(tokenized_docs_file)
        inverted_index = load_inverted_index(inverted_index_file)
        
        print("Tokenized Documents:")
        print(tokenized_docs[:2])  # Print first 2 documents for brevity
        
        print("\nInverted Index:")
        print(dict(list(inverted_index.items())[:2]))  # Print first 2 entries for brevity
    
    except Exception as e:
        print(f"An error occurred: {e}")

test_file_loading()