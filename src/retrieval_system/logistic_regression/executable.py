from datasets import load_dataset, disable_caching
from transformers import BertConfig
from preprocessing import PreprocessingModule
from training import TrainingModuleV2
import numpy as np
import ssl
import os
import sys
# import sqlitecloud

# sys.path.insert(0, f"{os.getcwd()}")
# from src.indexing.pre_processing import calculate_idf, calculate_tf_idf, fetch_and_tokenize_documents
# from src.crawler.sql_utils import insert_document


def create_glove_embedding_map():
    if pipeline_config.embedding_type == "glove":
        embedding_map = {}
        with open(pipeline_config.dataset_save_path+pipeline_config.glove_file, encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                try:
                    coefs = list(np.asarray(values[1:], dtype=float))
                except ValueError:
                    continue
                embedding_map[word] = coefs
    return embedding_map


if __name__ == '__main__':
    CONFIG_SAVE_PATH = "config/retrieval_system/"
    pipeline_config = BertConfig.from_json_file(CONFIG_SAVE_PATH+"pipeline_config.json")
    training_config = BertConfig.from_json_file(CONFIG_SAVE_PATH+f"{pipeline_config.model}-training_config.json")
    model_config = BertConfig.from_json_file(CONFIG_SAVE_PATH+f"{pipeline_config.model}-model_config.json")


    # dataset loading - replace with databank retrieval!
    DATASET_SIZE = training_config.dataset_size

    msmarco = load_dataset(
        "microsoft/ms_marco", "v1.1", split=f"train[:{DATASET_SIZE}]", verification_mode="no_checks"
    ).flatten()

    # try:
    #     _create_unverified_https_context = ssl._create_unverified_context
    # except AttributeError:
    #     pass
    # else:
    #     ssl._create_default_https_context = _create_unverified_https_context

    # api_key = "xZXTNaxWuKM6ryHCVELzSVnT3KC3AubraCDuwFyxKJ4"
    # db_url = "cyd2d2juiz.sqlite.cloud:8860"
    # db_name = "documents"

    # tokenized_docs_from_db = fetch_and_tokenize_documents(api_key, db_url, db_name)

    # if tokenized_docs_from_db:
    #     tf_idf_scores = calculate_tf_idf(tokenized_docs_from_db)
    #     idf_scores = calculate_idf(tokenized_docs_from_db)

    #     conn = sqlitecloud.connect(f"sqlitecloud://{db_url}?apikey={api_key}")
    #     conn.execute(f"USE DATABASE {db_name}")
    #     cursor = conn.cursor()

    #     #create_processed_documents_table(cursor)
    #     documents_from_db = get_all_documents(cursor)
    #     #store_processed_documents_and_tf_idf(cursor, documents_from_db, tokenized_docs_from_db, tf_idf_scores, idf_scores)

    #     conn.commit()
    #     conn.close()

    # print(tokenized_docs_from_db[:10])

    # exit()


    print("loading embedding map...")
    embed_map = None
    if pipeline_config.load_dataset_from_disk or pipeline_config.embedding_type != "glove":
        print(" - skipped")
    else:
        embed_map = create_glove_embedding_map()
        print(" - done")


    # tokenization + vocab creation
    PM = PreprocessingModule(
        training_config,
        model_config,
        pipeline_config
    )
    preprocessed_dataset = PM.execute(msmarco, embed_map)

    TM = TrainingModuleV2(
        training_config,
        model_config,
        pipeline_config
    )
    TM.execute(preprocessed_dataset)
