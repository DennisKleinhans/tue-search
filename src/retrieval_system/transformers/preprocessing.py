from module_classes import ProcessingModule
from nltk.tokenize import wordpunct_tokenize
from datasets import Dataset, disable_caching, load_from_disk
import numpy as np
from time import time
import json


def fast_multi_insert(list, indices, objects):
    """indices need to be sorted in ascending order!"""
    result = list
    tmp = []
    start = time()
    for num_indices_already_inserted, idx in enumerate(indices):
        interval = time() - start
        if interval > 0: # div by zero ;)
            print(" Inserting {:>10,d} rows, {:>5.2f} rows/sec".format(num_indices_already_inserted+1,(num_indices_already_inserted+1)/interval), end="\r")
        tmp.extend(result[:idx+num_indices_already_inserted])
        tmp.append(objects[num_indices_already_inserted])
        tmp.extend(result[idx+num_indices_already_inserted:])
        result = tmp
        tmp = []
    return result

def preprocess(string):
    return wordpunct_tokenize(string.lower())
    
def get_random_document(query_not_to_match, dataset):
    idx = np.random.randint(0, len(dataset))
    while query_not_to_match == dataset["query"][idx]:
        idx = np.random.randint(0, len(dataset))
    docs = dataset["passages.passage_text"][idx]
    return preprocess(docs[np.random.randint(0, len(docs))])

def get_vocab_encoding(token, vocab):
    result = vocab["<UNK>"]
    try:
        result = vocab[token]
    except KeyError:
        pass
    return result


class PreprocessingModule(ProcessingModule):
    def __init__(self, train_config, model_config, pipeline_config) -> None:
        super().__init__(train_config, model_config)
        self.pipeline_config = pipeline_config

        self.VOCAB = {
            "<PAD>": 0,
            "<UNK>": 1
        }

        # FF
        self.TOKENIZED_TEXT = []
        self.LABELS = []
        # SDE
        self.SDE_QUERIES = []
        self.SDE_DOCUMENTS = []
        self.SDE_TARGETS = []

        disable_caching()
    
    def __FF_process_batch(self, batch):
        for i in range(len(batch)):
            for document in batch["passages.passage_text"][i]:
                tokens = preprocess(" ".join([batch["query"][i], document]))
                self.TOKENIZED_TEXT.append(tokens)

                for token in tokens:
                    if token not in self.VOCAB:
                        self.VOCAB[token] = len(self.VOCAB)

            for label in batch["passages.is_selected"][i]:
                self.LABELS.append(label)

        return {}

    def __SDE_process_batch(self, batch):
        for i in range(len(batch)):
            query = batch["query"][i]

            # skip queries that have no ground truth documents
            if sum(batch["passages.is_selected"][i]) == 0:
                continue

            for doc_idx, document in enumerate(batch["passages.passage_text"][i]):
                tokens = preprocess(document)

                # update containers for generator
                self.SDE_QUERIES.append(preprocess(query))
                self.SDE_DOCUMENTS.append(tokens)
                self.SDE_TARGETS.append(batch["passages.is_selected"][i][doc_idx])
                    
                # update vocab
                for token in tokens:
                    if token not in self.VOCAB:
                        self.VOCAB[token] = len(self.VOCAB)
                    
        return {}
    
    def __SDE_pad_containers_to_batch_size(self, dataset):
        current_query = self.SDE_QUERIES[0]
        num_docs_per_this_query = 0
        # start = time()
        i = 0
        while True:
            try:
                tmp = self.SDE_QUERIES[i]
            except IndexError:
                break

            # interval = time() - start
            # if interval > 0: # div by zero ;)
            #     print(" Reading {:>10,d} rows, {:>5.2f} rows/sec".format(i+1,(i+1)/interval), end="\r")

            # check for switching query
            if current_query != self.SDE_QUERIES[i]:
                next_query = self.SDE_QUERIES[i]

                assert self.train_config.batch_size >= num_docs_per_this_query
                num_entries_to_add = self.train_config.batch_size-num_docs_per_this_query

                insertion_indices = [r+i for r in range(num_entries_to_add)]
                insertion_queries = [current_query for _ in range(num_entries_to_add)]
                insertion_documents = [get_random_document(current_query, dataset) for _ in range(num_entries_to_add)]
                insertion_targets = np.zeros(num_entries_to_add)

                self.SDE_QUERIES = fast_multi_insert(self.SDE_QUERIES, insertion_indices, insertion_queries)
                self.SDE_DOCUMENTS = fast_multi_insert(self.SDE_DOCUMENTS, insertion_indices, insertion_documents)
                self.SDE_TARGETS = fast_multi_insert(self.SDE_TARGETS, insertion_indices, insertion_targets)

                # added_entries = 0
                # while num_docs_per_this_query < self.train_config.batch_size:
                #     interval = time() - start
                #     if interval > 0: # div by zero ;)
                #         print(" Adding {:>10,d} rows, {:>5.2f} rows/sec".format(added_entries+1,(added_entries+1)/interval), end="\r")

                #     SDE_QUERIES.insert(i, current_query)
                #     SDE_DOCUMENTS.insert(i, _get_random_document(current_query))
                #     SDE_TARGETS.insert(i, 0)

                #     num_docs_per_this_query += 1
                #     added_entries += 1

                current_query = next_query
                i += num_entries_to_add  # resume from next query starting entry
                num_docs_per_this_query = 0 # reset doc counter
            else:
                num_docs_per_this_query += 1
                i += 1
        return
    
    def __gen_FF_preprocessed_dataset(self):
        for i in range(len(self.TOKENIZED_TEXT)):
            yield {"text": self.TOKENIZED_TEXT[i], "label": self.LABELS[i]}

    def __gen_SDE_preprocessed_dataset(self):
        for i in range(len(self.SDE_QUERIES)):
            yield {"query": self.SDE_QUERIES[i], "document": self.SDE_DOCUMENTS[i], "target": self.SDE_TARGETS[i]}

    def execute(self, dataset):
        print("preprocessing dataset...")

        BATCH_PROCESSING = {
            "FF": self.__FF_process_batch,
            "SDE": self.__SDE_process_batch,
        }

        GENERATOR = {
            "FF": self.__gen_FF_preprocessed_dataset,
            "SDE": self.__gen_SDE_preprocessed_dataset
        }

        if self.pipeline_config.load_dataset_from_disk:
            preprocessed_dataset = load_from_disk(self.pipeline_config.dataset_save_path+f"{self.pipeline_config.model}-preprocessed_dataset_bs{self.train_config.batch_size}")
            with open(self.pipeline_config.dataset_save_path+f"{self.pipeline_config.model}-vocab.json", "r", encoding="utf-8") as fs:
                self.VOCAB = json.load(fs)
                fs.close()
            print(" - loaded from disk")
        else:
            dataset.map(
                BATCH_PROCESSING[self.pipeline_config.model],
                batched=True
            )
            if self.train_config.batch_padding and self.train_config.batch_size >= 10:      
                print(f" batch padding to size {self.train_config.batch_size}...")
                self.__SDE_pad_containers_to_batch_size(dataset)
                print(" done")
                print("SDE_QUERIES")
                [print(f"{i} {q}") for i, q in enumerate(self.SDE_QUERIES[:self.train_config.batch_size*2])]
                print("SDE_DOCUMENTS")
                [print(f"{i} {q}") for i, q in enumerate(self.SDE_DOCUMENTS[:self.train_config.batch_size*2])]
                print("SDE_TARGETS") 
                [print(f"{i} {q}") for i, q in enumerate(self.SDE_TARGETS[:self.train_config.batch_size*2])]
                exit()

            preprocessed_dataset = Dataset.from_generator(GENERATOR[self.pipeline_config.model])

            preprocessed_dataset.save_to_disk(self.pipeline_config.dataset_save_path+f"{self.pipeline_config.model}-preprocessed_dataset_bs{self.train_config.batch_size}")
            with open(self.pipeline_config.dataset_save_path+f"{self.pipeline_config.model}-vocab.json", "w", encoding="utf-8") as fs:
                json.dump(self.VOCAB, fs, indent=4)
                fs.close()
            print(" - done")

        return preprocessed_dataset, self.VOCAB