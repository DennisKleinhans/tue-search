from module_classes import ProcessingModule
import os
from preprocessing import get_vocab_encoding
import json
import numpy as np
from datasets import load_from_disk
from time import time


class EmbeddingModule(ProcessingModule):
    def __init__(self, train_config, model_config, pipeline_config) -> None:
        super().__init__(train_config, model_config)
        self.pipeline_config = pipeline_config

        self.embedding_map = None

    
    def __create_glove_embedding_map(self):
        if self.pipeline_config.embedding_type == "glove":
            self.embedding_map = {}
            with open(self.pipeline_config.dataset_save_path+self.pipeline_config.glove_file, encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    try:
                        coefs = list(np.asarray(values[1:], dtype=float))
                    except ValueError:
                        continue
                    self.embedding_map[word] = coefs


    def __FF_embed_batch(self, batch, vocab):
        embeddings = [
            [get_vocab_encoding(token, vocab) for token in document][:self.model_config.max_position_embeddings]  # truncation
            for document in batch["text"]
        ]
        embeddings = [
            embed + [0] * (self.model_config.max_position_embeddings - len(embed)) for embed in embeddings  # padding
        ]
        return {"text": embeddings, "label": batch["label"]}


    def __SDE_embed_batch(self, batch, vocab):
        qry_embeddings = [
            [get_vocab_encoding(token, vocab) for token in query][:self.model_config.max_position_embeddings]  # truncation
            for query in batch["query"]
        ]
        qry_embeddings = [
            embed + [0] * (self.model_config.max_position_embeddings - len(embed)) for embed in qry_embeddings  # padding
        ]
        doc_embeddings = [
            [get_vocab_encoding(token, vocab) for token in document][:self.model_config.max_position_embeddings]  # truncation
            for document in batch["document"]
        ]
        doc_embeddings = [
            embed + [0] * (self.model_config.max_position_embeddings - len(embed)) for embed in doc_embeddings  # padding
        ]
        return {"query": qry_embeddings, "document": doc_embeddings, "target": batch["target"]}
    

    def __LR_embed_batch(self, batch):
        if self.pipeline_config.embedding_type == "glove":
            qry_embeddings = []
            doc_embeddings = []
            qry_tokens = []
            doc_tokens = []
            targets = []
            for i in range(len(batch)):
                query = batch["query"][i]
                embed = self.__get_glove_embed(query)
                qry_embeddings.append(embed)

                document = batch["document"][i]
                embed = self.__get_glove_embed(document)
                doc_embeddings.append(embed)

                targets.append(batch["target"][i])
                qry_tokens.append(query)
                doc_tokens.append(document)
        else:
            raise ValueError(f"LR: embedding type '{self.pipeline_config.embedding_type}' is not supported.")
        return {
            "query": qry_embeddings,
            "document": doc_embeddings,
            "target": targets,
            "query_tokens": qry_tokens,
            "document_tokens": doc_tokens
        }
    

    def __get_glove_embed(self, tokens):
        embeds = []
        for t in tokens:
            try:
                embeds.append(self.embedding_map[t])
            except KeyError:
                embeds.append([0.0])
        # return np.mean(embeds, axis=0) # ?!
        return embeds


    def execute(self, preprocessed_dataset, vocab):

        print("loading embedding map...")
        if self.pipeline_config.embedding_type == "glove":
            self.__create_glove_embedding_map()
            print(" - done")
        else:
            print(" - skipped because glove is not used")
        

        print("creating text embeddings...")

        EMBEDDING = {
            "FF": lambda batch: self.__FF_embed_batch(batch, vocab),
            "SDE": lambda batch: self.__SDE_embed_batch(batch, vocab),
            "LR": self.__LR_embed_batch
        }

        dataset_savename = f"{self.pipeline_config.dataset_save_path}{self.pipeline_config.model}-embedded_dataset_bs{self.train_config.batch_size}_embed-{self.pipeline_config.embedding_type}"
        if self.train_config.batch_padding:
            dataset_savename += "_padded"

        if self.pipeline_config.load_dataset_from_disk:
            embedded_dataset = load_from_disk(dataset_savename)
            print(" - loaded from disk")
        else:
            embedded_dataset = preprocessed_dataset.map(
                self.__LR_embed_batch,
                batched=True
            )
            embedded_dataset.save_to_disk(dataset_savename)
            print(" - done")
            

        print("converting dataset to numpy int32...")
        if self.pipeline_config.model == "FF":
            text = np.array(embedded_dataset["text"], dtype="int32")
            label = np.array(embedded_dataset["label"], dtype="int32")
            print(" - done")
            return text, label
        elif self.pipeline_config.model == "SDE":
            query = np.array(embedded_dataset["query"], dtype="int32")
            document = np.array(embedded_dataset["document"], dtype="int32")
            target = np.array(embedded_dataset["target"], dtype="int32")
            print(" - done")
            return query, document, target
        elif self.pipeline_config.model == "LR":
            return embedded_dataset
        else:
            raise ValueError(f"Model {self.pipeline_config.model} is not implemented!")