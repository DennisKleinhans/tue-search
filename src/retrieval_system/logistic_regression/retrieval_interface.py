from datasets import load_dataset
from transformers import BertConfig
import numpy as np
import sys
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

sys.path.insert(0, f"{os.getcwd()}")
from src.retrieval_system.logistic_regression.preprocessing import PreprocessingModule, preprocess
from src.retrieval_system.logistic_regression.training import TrainingModuleV2


class RetrievalSystemInterface():
    def __init__(self) -> None:
        CONFIG_SAVE_PATH = "config/retrieval_system/"

        self.pipeline_config = BertConfig.from_json_file(CONFIG_SAVE_PATH+"pipeline_config.json")
        self.training_config = BertConfig.from_json_file(CONFIG_SAVE_PATH+f"{self.pipeline_config.model}-training_config.json")
        self.model_config = BertConfig.from_json_file(CONFIG_SAVE_PATH+f"{self.pipeline_config.model}-model_config.json")

        self.PM = PreprocessingModule(
            self.training_config,
            self.model_config,
            self.pipeline_config
        )
        self.TM = TrainingModuleV2(
            self.training_config,
            self.model_config,
            self.pipeline_config
        )

    def __create_glove_embedding_map(self):
        if self.pipeline_config.embedding_type == "glove":
            embedding_map = {}
            with open(self.pipeline_config.dataset_save_path+self.pipeline_config.glove_file, encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    try:
                        coefs = list(np.asarray(values[1:], dtype=float))
                    except ValueError:
                        continue
                    embedding_map[word] = coefs
        return embedding_map


    def train_retrieval_system(self, dataset=None):
        if dataset is None:
            print("train_retrieval_system - WARNING: No training dataset provided. Training with default training set.")
            dataset = load_dataset(
                "microsoft/ms_marco", "v1.1", split=f"train[:{self.training_config.dataset_size}]", verification_mode="no_checks"
            ).flatten().rename_columns({
                "passages.passage_text": "document",
                "passages.is_selected": "label"
            })
            dataset = dataset.map(
                lambda batch: {
                    "query": batch["query"],
                    "document": [
                        batch["document"][j] 
                        for j in range(len(batch["document"]))
                    ],
                    "label": batch["label"]
                },
                batched=False,
                remove_columns=dataset.column_names
            )
        else:
            raise NotImplementedError("Using a custom dataset is no longer supported.")

        print("loading embedding map...")
        embed_map = None
        if self.pipeline_config.load_dataset_from_disk or self.pipeline_config.embedding_type != "glove":
            print(" - skipped")
        else:
            embed_map = self.__create_glove_embedding_map(self.pipeline_config)
            print(" - done")


        # tokenization + vocab creation
        preprocessed_dataset = self.PM.execute(dataset, embed_map)

        # training + evaluation
        self.TM.execute(preprocessed_dataset)


    def retrieve_ranking(self, query, preprocessing_results):
        return self.TM.retrieve(query, preprocessing_results)
