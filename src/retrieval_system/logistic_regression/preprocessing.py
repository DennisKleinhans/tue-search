from nltk.tokenize import wordpunct_tokenize
from datasets import Dataset, disable_caching, load_from_disk
import numpy as np
from time import time
import sys
import os

sys.path.insert(0, f"{os.getcwd()}")
from src.retrieval_system.logistic_regression.module_classes import ProcessingModule


def fast_multi_insert(list, idx, objects):
    """insertion of multiple elements in O(n+m)"""
    result = []
    result.extend(list[:idx])
    result.extend(objects)
    result.extend(list[idx:])
    return result


def preprocess(string):
    return wordpunct_tokenize(string.lower())


def get_glove_embed(tokens, embed_map):
    embed_size = 0
    for key in embed_map:
        embed_size = len(embed_map[key])
        break
    embeds = []
    for t in tokens:
        try:
            embeds.append(embed_map[t])
        except KeyError:
            embeds.append([0.0]*embed_size)
    # return np.mean(embeds, axis=0) # ?!
    return embeds


class PreprocessingModule(ProcessingModule):
    def __init__(self, train_config, model_config, pipeline_config) -> None:
        super().__init__(train_config, model_config)
        self.pipeline_config = pipeline_config
        disable_caching()

        self.pad_token = self.train_config.pad_token
        self.eos_token = self.train_config.eos_token

        self.embedding_map = None

        self.QUERIES = []
        self.DOCUMENTS = []
        self.QUERIES_EMBEDS = []
        self.DOCUMENTS_EMBEDS = []
        self.TARGETS = []
        # batch padding
        self.DOCUMENTS_BY_QUERY_INDEX = []
        self.DOCUMENT_EMBEDS_BY_QUERY_INDEX = []


    def __process_batch(self, batch, batched=True):
        if batched:
            for i in range(len(batch)):
                # skip queries that have no ground truth documents
                if sum(batch["label"][i]) == 0:
                    continue

                # query = batch["query"][i]
                # pp_query = preprocess(query)
                pp_query = batch["query"][i]
                pp_query = pp_query[:self.train_config.tokenizer_max_length] # truncation
                pp_query = pp_query + [self.pad_token]*((self.train_config.tokenizer_max_length-len(pp_query))-1) + [self.eos_token] # padding

                all_tokens = []
                all_tokens_embeds = []
                for doc_idx, document in enumerate(batch["document"][i]):
                    # pp_document = preprocess(document)
                    pp_document = document
                    pp_document = pp_document[:self.train_config.tokenizer_max_length] # truncation
                    pp_document = pp_document + [self.pad_token]*((self.train_config.tokenizer_max_length-len(pp_document))-1) + [self.eos_token] # padding

                    # update containers for generator
                    qry_embeds = get_glove_embed(pp_query, self.embedding_map)
                    doc_embeds = get_glove_embed(pp_document, self.embedding_map)

                    all_tokens.append(pp_document)
                    all_tokens_embeds.append(doc_embeds)

                    self.QUERIES.append(pp_query)
                    self.DOCUMENTS.append(pp_document)
                    self.QUERIES_EMBEDS.append(qry_embeds)
                    self.DOCUMENTS_EMBEDS.append(doc_embeds)
                    self.TARGETS.append(float(batch["label"][i][doc_idx]))

                self.DOCUMENTS_BY_QUERY_INDEX.append(all_tokens)
                self.DOCUMENT_EMBEDS_BY_QUERY_INDEX.append(all_tokens_embeds)
        else:
            # skip queries that have no ground truth documents
            if sum(batch["label"]) == 0:
                return {}

            # query = batch["query"]
            # pp_query = preprocess(query)
            pp_query = batch["query"]
            pp_query = pp_query[:self.train_config.tokenizer_max_length] # truncation
            pp_query = pp_query + [self.pad_token]*((self.train_config.tokenizer_max_length-len(pp_query))-1) + [self.eos_token] # padding

            all_tokens = []
            all_tokens_embeds = []
            for doc_idx, document in enumerate(batch["document"]):
                # pp_document = preprocess(document)
                pp_document = document
                pp_document = pp_document[:self.train_config.tokenizer_max_length] # truncation
                pp_document = pp_document + [self.pad_token]*((self.train_config.tokenizer_max_length-len(pp_document))-1) + [self.eos_token] # padding

                # update containers for generator
                if self.pipeline_config.embedding_type == "glove":
                    qry_embeds = get_glove_embed(pp_query, self.embedding_map)
                    doc_embeds = get_glove_embed(pp_document, self.embedding_map)
                else: # quick and dirty solution ;)
                    qry_embeds = []
                    doc_embeds = []

                all_tokens.append(pp_document)
                all_tokens_embeds.append(doc_embeds)

                self.QUERIES.append(pp_query)
                self.DOCUMENTS.append(pp_document)
                self.QUERIES_EMBEDS.append(qry_embeds)
                self.DOCUMENTS_EMBEDS.append(doc_embeds)
                self.TARGETS.append(float(batch["label"][doc_idx]))

            self.DOCUMENTS_BY_QUERY_INDEX.append(all_tokens)
            self.DOCUMENT_EMBEDS_BY_QUERY_INDEX.append(all_tokens_embeds)

        return {}
    

    def __get_random_document(self, query_idx_not_to_match):
        q_idx = np.random.randint(0, len(self.DOCUMENTS_BY_QUERY_INDEX))
        while q_idx == query_idx_not_to_match:
            q_idx = np.random.randint(0, len(self.DOCUMENTS_BY_QUERY_INDEX))
        docs = self.DOCUMENTS_BY_QUERY_INDEX[q_idx]
        d_idx = np.random.randint(0, len(docs))
        return docs[d_idx], self.DOCUMENT_EMBEDS_BY_QUERY_INDEX[q_idx][d_idx]
    

    def __pad_containers_to_batch_size(self):
        current_query = self.QUERIES[0]
        current_query_embeds = self.QUERIES_EMBEDS[0]
        num_docs_per_this_query = 0
        start = time()
        i = 0
        while True:
            try:
                tmp = self.QUERIES[i]
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
            if current_query != self.QUERIES[i]:
                next_query = self.QUERIES[i]
                next_query_embeds = self.QUERIES_EMBEDS[i]
                
                if self.train_config.batch_size < num_docs_per_this_query:
                    # reduce to batch size
                    while self.train_config.batch_size < num_docs_per_this_query:
                        rnd_idx = np.random.randint(i-num_docs_per_this_query, i)
                        if self.TARGETS[rnd_idx] != 1:
                            del self.QUERIES[rnd_idx]
                            del self.DOCUMENTS[rnd_idx]
                            del self.QUERIES_EMBEDS[rnd_idx]
                            del self.DOCUMENTS_EMBEDS[rnd_idx]
                            del self.TARGETS[rnd_idx]
                            i -= 1
                            num_docs_per_this_query -= 1
                else:
                    # pad to batch size
                    num_entries_to_add = self.train_config.batch_size-num_docs_per_this_query

                    insertion_index = i
                    insertion_queries = []
                    insertion_documents = []
                    insertion_queries_embeds = []
                    insertion_documents_embeds = []
                    for _ in range(num_entries_to_add):
                        insertion_queries.append(current_query)
                        insertion_queries_embeds.append(current_query_embeds)

                        d, e = self.__get_random_document(i-1)
                        insertion_documents.append(d)
                        insertion_documents_embeds.append(e)

                    insertion_targets = np.zeros(num_entries_to_add)

                    self.QUERIES = fast_multi_insert(self.QUERIES, insertion_index, insertion_queries)
                    self.DOCUMENTS = fast_multi_insert(self.DOCUMENTS, insertion_index, insertion_documents)
                    self.QUERIES_EMBEDS = fast_multi_insert(self.QUERIES_EMBEDS, insertion_index, insertion_queries_embeds)
                    self.DOCUMENTS_EMBEDS = fast_multi_insert(self.DOCUMENTS_EMBEDS, insertion_index, insertion_documents_embeds)
                    self.TARGETS = fast_multi_insert(self.TARGETS, insertion_index, insertion_targets)

                    num_docs_per_this_query += num_entries_to_add
                    i += num_entries_to_add  # resume from next query starting entry

                # ensure only one ground truth per batch
                bt_lower_bound = i-num_docs_per_this_query
                batch_targets = self.TARGETS[bt_lower_bound:i]
                gt_indices = [idx+bt_lower_bound for idx, e in enumerate(batch_targets) if e == 1]
                if len(gt_indices) > 1:
                    for idx in gt_indices[1:]: # keep only the first ground truth
                        self.TARGETS[idx] = 0.0

                assert self.train_config.batch_size == num_docs_per_this_query # correct batch size
                assert np.sum(self.TARGETS[bt_lower_bound:i]) == 1.0 # one ground truth per batch
                assert next_query == self.QUERIES[i] # index is still correct

                current_query = next_query
                current_query_embeds = next_query_embeds
                num_docs_per_this_query = 0 # reset doc counter
            else:
                num_docs_per_this_query += 1
                i += 1
        print()
        return

    def __gen_preprocessed_dataset(self):
        for i in range(len(self.QUERIES)):
            yield {"query": self.QUERIES[i], "document": self.DOCUMENTS[i], "query_embeds": self.QUERIES_EMBEDS[i], "document_embeds": self.DOCUMENTS_EMBEDS[i], "target": self.TARGETS[i]}

    def execute(self, dataset, embed_map):
        if self.pipeline_config.embedding_type == "glove":
            self.embedding_map = embed_map

        print("preprocessing dataset...")
        dataset_savename = f"{self.pipeline_config.dataset_save_path}preprocessed_dataset_bs{self.train_config.batch_size}_embed-{self.pipeline_config.embedding_type}"
        if self.pipeline_config.embedding_type == "glove":
            dataset_savename += "-".join(self.pipeline_config.glove_file.lstrip("glove").rstrip(".txt").split("."))
        if self.train_config.batch_padding:
            dataset_savename += "_padded"
        dataset_savename += f"_tml{self.train_config.tokenizer_max_length}"

        dataset_savename += f"_1.1_train-{self.train_config.dataset_size}"

        if self.pipeline_config.load_dataset_from_disk:
            preprocessed_dataset = load_from_disk(dataset_savename)
            print(" - loaded from disk")
        else:
            batched = False
            dataset.map(
                lambda batch: self.__process_batch(batch, batched),
                batched=batched
            )
            if self.train_config.batch_padding:      
                print(f" batch padding to size {self.train_config.batch_size}...")
                self.__pad_containers_to_batch_size()
                print(f" padded dataset size: {len(self.QUERIES)}")
                print("  - done")

            print(" generating preprocessed dataset...")
            preprocessed_dataset = Dataset.from_generator(self.__gen_preprocessed_dataset)

            preprocessed_dataset.save_to_disk(dataset_savename)
            print(" - done")

        return preprocessed_dataset