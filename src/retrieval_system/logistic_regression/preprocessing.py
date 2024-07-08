from module_classes import ProcessingModule
from nltk.tokenize import wordpunct_tokenize
from datasets import Dataset, disable_caching, load_from_disk
import numpy as np
from time import time
import json


def fast_multi_insert(list, idx, objects):
    """insertion of multiple elements in O(n+m)"""
    result = []
    result.extend(list[:idx])
    result.extend(objects)
    result.extend(list[idx:])
    return result

def preprocess(string):
    return wordpunct_tokenize(string.lower())

def get_vocab_encoding(token, vocab):
    result = vocab["<UNK>"]
    try:
        result = vocab[token]
    except KeyError:
        pass
    return result

def revert_vocab_encoding(encoding_idx, vocab):
    result = "<UNK>"
    try:
        keys = list(vocab.keys())
        result = keys[encoding_idx]
    except IndexError:
        pass
    return result

class PreprocessingModule(ProcessingModule):
    def __init__(self, train_config, model_config, pipeline_config) -> None:
        super().__init__(train_config, model_config)
        self.pipeline_config = pipeline_config
        self.pad_token = "<PAD>"
        self.eos_token = "<EOS>"

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
        # batch padding (SDE only)
        self.DOCUMENTS_BY_QUERY_INDEX = []

        disable_caching() # important for keeping the vocab fresh


    def __SDE_process_batch(self, batch):
        for i in range(len(batch)):
            # skip queries that have no ground truth documents
            if sum(batch["passages.is_selected"][i]) == 0:
                continue

            query = batch["query"][i]
            pp_query = preprocess(query)
            pp_query = pp_query[:self.train_config.tokenizer_max_length] # truncation
            pp_query = pp_query + [self.pad_token]*((self.train_config.tokenizer_max_length-len(pp_query))-1) + [self.eos_token] # padding

            all_tokens = []
            for doc_idx, document in enumerate(batch["passages.passage_text"][i]):
                pp_document = preprocess(document)
                pp_document = pp_document[:self.train_config.tokenizer_max_length] # truncation
                pp_document = pp_document + [self.pad_token]*((self.train_config.tokenizer_max_length-len(pp_document))-1) + [self.eos_token] # padding

                all_tokens.append(pp_document)

                # update containers for generator
                self.SDE_QUERIES.append(pp_query)
                self.SDE_DOCUMENTS.append(pp_document)
                self.SDE_TARGETS.append(float(batch["passages.is_selected"][i][doc_idx]))
                    
                # update vocab
                for token in pp_document:
                    if token not in self.VOCAB:
                        self.VOCAB[token] = len(self.VOCAB)

            self.DOCUMENTS_BY_QUERY_INDEX.append(all_tokens)
                    
        return {}
    

    def __get_random_document(self, query_idx_not_to_match):
        q_idx = np.random.randint(0, len(self.DOCUMENTS_BY_QUERY_INDEX))
        while q_idx == query_idx_not_to_match:
            q_idx = np.random.randint(0, len(self.DOCUMENTS_BY_QUERY_INDEX))
        docs = self.DOCUMENTS_BY_QUERY_INDEX[q_idx]
        d_idx = np.random.randint(0, len(docs))
        return docs[d_idx]
    

    def __SDE_pad_containers_to_batch_size(self):
        current_query = self.SDE_QUERIES[0]
        num_docs_per_this_query = 0
        start = time()
        i = 0
        while True:
            try:
                tmp = self.SDE_QUERIES[i]
            except IndexError:
                break

            # debug
            # if i > 35:
            #     for l in [self.SDE_QUERIES, self.SDE_DOCUMENTS, self.SDE_TARGETS]:
            #         print(l[:35])
            #     exit()

            interval = time() - start
            if (i + 1) % 10 == 0 and interval > 0:
                print(" Reading {:>10,d} rows, {:>5.2f} rows/sec ".format(i+1,(i+1)/interval), end="\r")

            # check for switching query
            if current_query != self.SDE_QUERIES[i]:
                next_query = self.SDE_QUERIES[i]
                
                if self.train_config.batch_size < num_docs_per_this_query:
                    # reduce to batch size
                    while self.train_config.batch_size < num_docs_per_this_query:
                        rnd_idx = np.random.randint(i-num_docs_per_this_query, i)
                        if self.SDE_TARGETS[rnd_idx] != 1:
                            del self.SDE_QUERIES[rnd_idx]
                            del self.SDE_DOCUMENTS[rnd_idx]
                            del self.SDE_TARGETS[rnd_idx]
                            i -= 1
                            num_docs_per_this_query -= 1
                else:
                    # pad to batch size
                    num_entries_to_add = self.train_config.batch_size-num_docs_per_this_query

                    insertion_index = i
                    insertion_queries = []
                    insertion_documents = []
                    for _ in range(num_entries_to_add):
                        insertion_queries.append(current_query)
                        insertion_documents.append(self.__get_random_document(i-1))
                    insertion_targets = np.zeros(num_entries_to_add)

                    self.SDE_QUERIES = fast_multi_insert(self.SDE_QUERIES, insertion_index, insertion_queries)
                    self.SDE_DOCUMENTS = fast_multi_insert(self.SDE_DOCUMENTS, insertion_index, insertion_documents)
                    self.SDE_TARGETS = fast_multi_insert(self.SDE_TARGETS, insertion_index, insertion_targets)

                    num_docs_per_this_query += num_entries_to_add
                    i += num_entries_to_add  # resume from next query starting entry

                # ensure only one ground truth per batch
                bt_lower_bound = i-num_docs_per_this_query
                batch_targets = self.SDE_TARGETS[bt_lower_bound:i]
                gt_indices = [idx+bt_lower_bound for idx, e in enumerate(batch_targets) if e == 1]
                if len(gt_indices) > 1:
                    for idx in gt_indices[1:]: # keep only the first ground truth
                        self.SDE_TARGETS[idx] = 0.0

                assert self.train_config.batch_size == num_docs_per_this_query # correct batch size
                assert np.sum(self.SDE_TARGETS[bt_lower_bound:i]) == 1.0 # one ground truth per batch
                assert next_query == self.SDE_QUERIES[i] # index is still correct

                current_query = next_query
                num_docs_per_this_query = 0 # reset doc counter
            else:
                num_docs_per_this_query += 1
                i += 1
        print()
        return

    def __gen_SDE_preprocessed_dataset(self):
        for i in range(len(self.SDE_QUERIES)):
            yield {"query": self.SDE_QUERIES[i], "document": self.SDE_DOCUMENTS[i], "target": self.SDE_TARGETS[i]}

    def execute(self, dataset):
        print("preprocessing dataset...")

        BATCH_PROCESSING = {
            "LR": self.__SDE_process_batch,
        }

        GENERATOR = {
            "LR": self.__gen_SDE_preprocessed_dataset,
        }

        dataset_savename = f"{self.pipeline_config.dataset_save_path}{self.pipeline_config.model}-preprocessed_dataset_bs{self.train_config.batch_size}_embed-{self.pipeline_config.embedding_type}"
        if self.pipeline_config.embedding_type == "glove":
            dataset_savename += "-".join(self.pipeline_config.glove_file.lstrip("glove").rstrip(".txt").split("."))
        if self.train_config.batch_padding:
            dataset_savename += "_padded"
        dataset_savename += f"_tml{self.train_config.tokenizer_max_length}"

        if self.pipeline_config.load_dataset_from_disk:
            preprocessed_dataset = load_from_disk(dataset_savename)
            with open(self.pipeline_config.dataset_save_path+f"{self.pipeline_config.model}-vocab.json", "r", encoding="utf-8") as fs:
                self.VOCAB = json.load(fs)
                fs.close()
            print(" - loaded from disk")
        else:
            dataset.map(
                BATCH_PROCESSING[self.pipeline_config.model],
                batched=True
            )
            if self.train_config.batch_padding:      
                print(f" batch padding to size {self.train_config.batch_size}...")
                self.__SDE_pad_containers_to_batch_size()
                print(f" padded dataset size: {len(self.SDE_QUERIES)}")
                print("  - done")

            preprocessed_dataset = Dataset.from_generator(GENERATOR[self.pipeline_config.model])

            preprocessed_dataset.save_to_disk(dataset_savename)
            with open(self.pipeline_config.dataset_save_path+f"{self.pipeline_config.model}-vocab.json", "w", encoding="utf-8") as fs:
                json.dump(self.VOCAB, fs, indent=4)
                fs.close()
            print(" - done")

        return preprocessed_dataset, self.VOCAB