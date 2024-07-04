from module_classes import ProcessingModule
from preprocessing import get_vocab_encoding
import numpy as np
from datasets import load_from_disk


class IndexEmbeddingModule(ProcessingModule):
    def __init__(self, train_config, model_config, pipeline_config) -> None:
        super().__init__(train_config, model_config)
        self.pipeline_config = pipeline_config

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

    def execute(self, preprocessed_dataset, vocab):
        print("creating text embeddings...")

        EMBEDDING = {
            "FF": lambda batch: self.__FF_embed_batch(batch, vocab),
            "SDE": lambda batch: self.__SDE_embed_batch(batch, vocab)
        }

        dataset_savename = f"{self.pipeline_config.dataset_save_path}{self.pipeline_config.model}-embedded_dataset_bs{self.train_config.batch_size}"
        if self.train_config.batch_padding:
            dataset_savename += "_padded"

        if self.pipeline_config.load_dataset_from_disk:
            embedded_dataset = load_from_disk(dataset_savename)
            print(" - loaded from disk")
        else:
            embedded_dataset = preprocessed_dataset.map(
                EMBEDDING[self.pipeline_config.model], 
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
        else:
            raise ValueError(f"Model {self.pipeline_config.model} is not implemented!")