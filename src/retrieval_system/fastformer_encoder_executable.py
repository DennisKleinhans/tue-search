from datasets import Dataset, load_dataset, disable_caching, load_from_disk
from transformers import BertConfig
from nltk.tokenize import wordpunct_tokenize
import numpy as np
from time import time
import torch
from Fastformer import Model
from SiameseDualEncoder import SDE
from scipy.stats import kendalltau, pearsonr
from sklearn.metrics import ndcg_score
import json

MODEL = "SDE" # FF or SDE

PERFORM_RETRIEVAL = True # FF only

LOAD_DATASET_FROM_DISK = False
DATASET_SAVE_PATH = "data/retrieval_system/"
CONFIG_SAVE_PATH = "config/retrieval_system/"
SAVE_PREFIX = f"{MODEL}-"

training_config = BertConfig.from_json_file(CONFIG_SAVE_PATH+SAVE_PREFIX+"training_config.json")
model_config = BertConfig.from_json_file(CONFIG_SAVE_PATH+SAVE_PREFIX+"model_config.json")

SELECTED_METRIC = "pcc"
METRICS = {
    "pcc": (lambda true, hat: pearsonr(true, hat).statistic, "Pearson correlation coefficient"),
    "ndcg10": (lambda true, hat: ndcg_score(true, hat, k=10), "Normalized Discounted Cumulative Gain @ 10"),
}


def preprocess(string):
    return wordpunct_tokenize(string.lower())


# dataset loading
disable_caching()  # needed for vocab generation!!!
msmarco = load_dataset(
    "microsoft/ms_marco", "v2.1", split="train", verification_mode="no_checks"
).flatten()


# tokenization + vocab creation
print("preprocessing dataset...")
VOCAB = {
    "<PAD>": 0,
    "<UNK>": 1
}

def get_vocab_encoding(token):
    result = VOCAB["<UNK>"]
    try:
        result = VOCAB[token]
    except KeyError:
        pass
    return result

# FF
TOKENIZED_TEXT = []
LABELS = []

def FF_process_batch(batch):
    for i in range(len(batch)):
        for document in batch["passages.passage_text"][i]:
            tokens = preprocess(" ".join([batch["query"][i], document]))
            TOKENIZED_TEXT.append(tokens)

            for token in tokens:
                if token not in VOCAB:
                    VOCAB[token] = len(VOCAB)

        for label in batch["passages.is_selected"][i]:
            LABELS.append(label)

    return {}

# SDE
SDE_QUERIES = []
SDE_DOCUMENTS = []
SDE_TARGETS = []

def SDE_process_batch(batch):
    for i in range(len(batch)):
        query = batch["query"][i]

        # skip queries that have no ground truth documents
        if sum(batch["passages.is_selected"][i]) == 0:
            continue

        for doc_idx, document in enumerate(batch["passages.passage_text"][i]):
            tokens = preprocess(document)

            # update containers for generator
            SDE_QUERIES.append(preprocess(query))
            SDE_DOCUMENTS.append(tokens)
            SDE_TARGETS.append(batch["passages.is_selected"][i][doc_idx])
            
            # update vocab
            for token in tokens:
                if token not in VOCAB:
                    VOCAB[token] = len(VOCAB)
            
    return {}

BATCH_PROCESSING = {
    "FF": FF_process_batch,
    "SDE": SDE_process_batch,
}

def gen_FF_preprocessed_dataset():
    for i in range(len(TOKENIZED_TEXT)):
        yield {"text": TOKENIZED_TEXT[i], "label": LABELS[i]}

def gen_SDE_preprocessed_dataset():
    for i in range(len(SDE_QUERIES)):
        yield {"query": SDE_QUERIES[i], "document": SDE_DOCUMENTS[i], "target": SDE_TARGETS[i]}

GENERATOR = {
    "FF": gen_FF_preprocessed_dataset,
    "SDE": gen_SDE_preprocessed_dataset
}

if LOAD_DATASET_FROM_DISK:
    preprocessed_dataset = load_from_disk(DATASET_SAVE_PATH+SAVE_PREFIX+f"preprocessed_dataset_bs{training_config.batch_size}")
    with open(DATASET_SAVE_PATH+SAVE_PREFIX+"vocab.json", "r", encoding="utf-8") as fs:
        VOCAB = json.load(fs)
        fs.close()
    print(" - loaded from disk")
else:
    msmarco.map(
        BATCH_PROCESSING[MODEL],
        batched=True
    )
    preprocessed_dataset = Dataset.from_generator(GENERATOR[MODEL])
    if training_config.batch_padding and training_config.batch_size >= 10:
        def SDE_pad_containers_to_batch_size():
            def _get_random_document(query_not_to_match):
                idx = np.random.randint(0, len(msmarco))
                while query_not_to_match == msmarco["query"][idx]:
                    idx = np.random.randint(0, len(msmarco))
                docs = msmarco["passages.passage_text"][idx]
                return docs[np.random.randint(0, len(docs))]

            current_query = SDE_QUERIES[0]
            num_docs_per_this_query = 0
            start = time()
            i = 0
            while True:
                try:
                    tmp = SDE_QUERIES[i]
                except IndexError:
                    break

                # debug
                if i >= 35:
                    return

                interval = time() - start
                if interval > 0: # div by zero ;)
                    print(f" Reading {i+1:>10,d} lines, {(i+1)/interval:>5.2f} queries/sec", end="\r")

                # check for switching query
                if current_query != SDE_QUERIES[i]:
                    print(f"i={i}: current_query: {current_query}")
                    next_query = SDE_QUERIES[i]
                    next_query_start_idx = i
                    print(f"i={i}: next_query: {next_query}")
                    added_entries = 0
                    while num_docs_per_this_query < training_config.batch_size:
                        print(num_docs_per_this_query)
                        SDE_QUERIES.insert(i, current_query)
                        SDE_DOCUMENTS.insert(i, _get_random_document(current_query))
                        SDE_TARGETS.insert(i, 0)
                        num_docs_per_this_query += 1
                        added_entries += 1
                    current_query = next_query
                    i = next_query_start_idx # resume from next query starting entry
                    print(f"new i: {i}")
                    num_docs_per_this_query = 0 # reset doc counter
                    continue

                num_docs_per_this_query += 1
                i += 1
            return
        
        print(f" batch padding to size {training_config.batch_size}...")
        SDE_pad_containers_to_batch_size()
        print(" done")
        print("SDE_QUERIES")
        [print(f"{i} {q}") for i, q in enumerate(SDE_QUERIES[:training_config.batch_size*2])]
        print("SDE_DOCUMENTS")
        [print(f"{i} {q}") for i, q in enumerate(SDE_DOCUMENTS[:training_config.batch_size*2])]
        print("SDE_TARGETS") 
        [print(f"{i} {q}") for i, q in enumerate(SDE_TARGETS[:training_config.batch_size*2])]
        exit()

    preprocessed_dataset.save_to_disk(DATASET_SAVE_PATH+SAVE_PREFIX+f"preprocessed_dataset_bs{training_config.batch_size}")
    with open(DATASET_SAVE_PATH+SAVE_PREFIX+"vocab.json", "w", encoding="utf-8") as fs:
        json.dump(VOCAB, fs, indent=4)
        fs.close()
    print(" - done")

# update model params
model_config.num_embeddings = len(VOCAB)
model_config.vocab_size = len(VOCAB)


# index embedding + padding
print("creating text embeddings...")
def FF_embed_batch(batch):
    embeddings = [
        [get_vocab_encoding(token) for token in document][:model_config.max_position_embeddings]  # truncation
        for document in batch["text"]
    ]
    embeddings = [
        embed + [0] * (model_config.max_position_embeddings - len(embed)) for embed in embeddings  # padding
    ]
    return {"text": embeddings, "label": batch["label"]}

def SDE_embed_batch(batch):
    qry_embeddings = [
        [get_vocab_encoding(token) for token in query][:model_config.max_position_embeddings]  # truncation
        for query in batch["query"]
    ]
    qry_embeddings = [
        embed + [0] * (model_config.max_position_embeddings - len(embed)) for embed in qry_embeddings  # padding
    ]
    doc_embeddings = [
        [get_vocab_encoding(token) for token in document][:model_config.max_position_embeddings]  # truncation
        for document in batch["document"]
    ]
    doc_embeddings = [
        embed + [0] * (model_config.max_position_embeddings - len(embed)) for embed in doc_embeddings  # padding
    ]
    return {"query": qry_embeddings, "document": doc_embeddings, "target": batch["target"]}

EMBEDDING = {
    "FF": FF_embed_batch,
    "SDE": SDE_embed_batch
}

if LOAD_DATASET_FROM_DISK:
    embedded_dataset = load_from_disk(DATASET_SAVE_PATH+SAVE_PREFIX+f"embedded_dataset_bs{training_config.batch_size}")
    print(" - loaded from disk")
else:
    embedded_dataset = preprocessed_dataset.map(
        EMBEDDING[MODEL], 
        batched=True
    )
    embedded_dataset.save_to_disk(DATASET_SAVE_PATH+SAVE_PREFIX+f"embedded_dataset_bs{training_config.batch_size}")
    print(" - done")

# size = 0
# for row in embedded_dataset:
#     if size < len(row["text"]):
#         size = len(row["text"])
# print(size)


print("converting dataset to numpy int32...")
if MODEL == "FF":
    text = np.array(embedded_dataset["text"], dtype="int32")
    label = np.array(embedded_dataset["label"], dtype="int32")
elif MODEL == "SDE":
    query = np.array(embedded_dataset["query"], dtype="int32")
    document = np.array(embedded_dataset["document"], dtype="int32")
    target = np.array(embedded_dataset["target"], dtype="int32")
else:
    raise ValueError(f"Model {MODEL} is not implemented!")
print(" - done")


print("creating split indices...")
if MODEL == "FF":
    index = np.arange(len(label))
elif MODEL == "SDE":
    index = np.arange(len(target))
else:
    raise ValueError(f"Model {MODEL} is not implemented!")

breakpoint = int(np.ceil(training_config.train_split_size * len(index)))
train_idx = index[:breakpoint]
if not training_config.batch_padding:
    np.random.shuffle(train_idx)
test_idx = index[breakpoint:]
print(" - done")


print(f"training {MODEL}...")
if MODEL == "FF":
    model = Model(model_config)
elif MODEL == "SDE":
    model = SDE(model_config)
else:
    raise ValueError(f"Model {MODEL} is not implemented!")

import torch.optim as optim
optimizer = optim.Adam(
    [{"params": model.parameters(), "lr": training_config.learning_rate}]
)

model.cuda()

if MODEL == "FF":
    for i in range(training_config.num_epochs):
        loss = 0.0
        metric_score = 0.0
        for num_batch in range(len(train_idx) // training_config.batch_size):

            log_ids = text[train_idx][
                num_batch * training_config.batch_size
                : num_batch * training_config.batch_size + training_config.batch_size,
                : training_config.tokenizer_max_length,
            ]
            targets = label[train_idx][
                num_batch * training_config.batch_size 
                : num_batch * training_config.batch_size + training_config.batch_size
            ]

            log_ids = torch.LongTensor(log_ids).cuda(0, non_blocking=True)
            targets = torch.FloatTensor(targets).cuda(0, non_blocking=True)
            bz_loss, y_hat = model(log_ids, targets)

            loss += bz_loss.data.float()

            this_score = METRICS[SELECTED_METRIC][0](
                targets.to("cpu").detach().numpy().tolist(),
                y_hat.to("cpu").detach().numpy().tolist()
            )
            # scipy metrics sometimes return nan
            if np.isnan(this_score):
                this_score = 0
            metric_score += this_score

            unified_loss = bz_loss
            optimizer.zero_grad()
            unified_loss.backward()
            optimizer.step()

            if num_batch % 100 == 0:
                print(
                    "[TRAIN SET] Epoch: {}, Samples: 0-{}, train_loss: {:.5f}, {}: {:.5f}".format(
                        i+1,
                        training_config.batch_size + (num_batch * training_config.batch_size), 
                        loss.data / (num_batch + 1), # mean over all batches processed so far
                        SELECTED_METRIC,
                        metric_score / (num_batch + 1) # mean over all batches processed so far
                    )
                )
        model.eval()
        y_hat_all = []
        loss2 = 0.0
        for num_batch in range(len(test_idx) // training_config.batch_size + 1):

            log_ids = text[test_idx][
                num_batch * training_config.batch_size 
                : num_batch * training_config.batch_size + training_config.batch_size, 
                : training_config.tokenizer_max_length
            ]
            targets = label[test_idx][
                num_batch * training_config.batch_size 
                : num_batch * training_config.batch_size + training_config.batch_size
            ]
            log_ids = torch.LongTensor(log_ids).cuda(0, non_blocking=True)
            targets = torch.FloatTensor(targets).cuda(0, non_blocking=True)

            bz_loss2, y_hat2 = model(log_ids, targets)

            loss2 += bz_loss2.data.float()
            y_hat_all += y_hat2.to("cpu").detach().numpy().tolist()

        y_true = label[test_idx]
        print("[TEST SET] {} after epoch {}: {:.5f} (loss: {:.5})\n".format(
            METRICS[SELECTED_METRIC][1],
            i+1,
            METRICS[SELECTED_METRIC][0](y_true, y_hat_all),
            loss2
        ))
        model.train()
elif MODEL == "SDE":
    for i in range(training_config.num_epochs):
        loss = 0.0
        metric_score = 0.0
        for num_batch in range(len(train_idx) // training_config.batch_size):

            qry_logids = query[train_idx][
                num_batch * training_config.batch_size
                : num_batch * training_config.batch_size + training_config.batch_size,
                : training_config.tokenizer_max_length,
            ]
            doc_logids = document[train_idx][
                num_batch * training_config.batch_size
                : num_batch * training_config.batch_size + training_config.batch_size,
                : training_config.tokenizer_max_length,
            ]
            targets = target[train_idx][
                num_batch * training_config.batch_size 
                : num_batch * training_config.batch_size + training_config.batch_size
            ]
            qry_logids = torch.LongTensor(qry_logids).cuda(0, non_blocking=True)
            doc_logids = torch.LongTensor(doc_logids).cuda(0, non_blocking=True)
            targets = torch.FloatTensor(targets).cuda(0, non_blocking=True)

            bz_loss, y_hat = model(qry_logids, doc_logids, targets)

            loss += bz_loss.data.float()

            this_score = METRICS[SELECTED_METRIC][0](
                targets.to("cpu").detach().numpy().tolist(),
                y_hat.to("cpu").detach().numpy().tolist()
            )
            # scipy metrics sometimes return nan
            if np.isnan(this_score):
                this_score = 0
            metric_score += this_score

            unified_loss = bz_loss
            optimizer.zero_grad()
            unified_loss.backward()
            optimizer.step()

            if num_batch % 100 == 0:
                print(
                    "[TRAIN SET] Epoch: {}, Samples: 0-{}, train_loss: {:.5f}, {}: {:.5f}".format(
                        i+1,
                        training_config.batch_size + (num_batch * training_config.batch_size), 
                        loss.data / (num_batch + 1), # mean over all batches processed so far
                        SELECTED_METRIC,
                        metric_score / (num_batch + 1) # mean over all batches processed so far
                    )
                )
        model.eval()
        y_hat_all = []
        loss2 = 0.0
        for num_batch in range(len(test_idx) // training_config.batch_size + 1):

            qry_logids = query[test_idx][
                num_batch * training_config.batch_size
                : num_batch * training_config.batch_size + training_config.batch_size,
                : training_config.tokenizer_max_length,
            ]
            doc_logids = document[test_idx][
                num_batch * training_config.batch_size
                : num_batch * training_config.batch_size + training_config.batch_size,
                : training_config.tokenizer_max_length,
            ]
            targets = target[test_idx][
                num_batch * training_config.batch_size 
                : num_batch * training_config.batch_size + training_config.batch_size
            ]
            qry_logids = torch.LongTensor(qry_logids).cuda(0, non_blocking=True)
            doc_logids = torch.LongTensor(doc_logids).cuda(0, non_blocking=True)
            targets = torch.FloatTensor(targets).cuda(0, non_blocking=True)

            bz_loss2, y_hat2 = model(qry_logids, doc_logids, targets)

            loss2 += bz_loss2.data.float()
            y_hat_all += y_hat2.to("cpu").detach().numpy().tolist()

        y_true = label[test_idx]
        print("[TEST SET] {} after epoch {}: {:.5f} (loss: {:.5})\n".format(
            METRICS[SELECTED_METRIC][1],
            i+1,
            METRICS[SELECTED_METRIC][0](y_true, y_hat_all),
            loss2
        ))
        model.train()
else:
    raise ValueError(f"Model {MODEL} is not implemented!")
print(" - done")


print("retrieving for a random query from all documents...")
if PERFORM_RETRIEVAL and (MODEL == "FF"):
    RANDOM_QUERY = msmarco["query"][np.random.randint(0, len(msmarco))]
    MATCHED_DOCUMENT = "NO_DOCUMENT_MATCHED_THE_QUERY"
    ONLY_DOCUMENTS = [] # for lookup of the retrieved documents' text
    DOCUMENT_TEXT = [] # global for constructing the embeddings
    TOP_K = 10 # num of retrieved docs

    def process_batch_text(batch):
        for i in range(len(batch)):
            for document in batch["passages.passage_text"][i]:
                tokens = preprocess(" ".join([RANDOM_QUERY, document]))
                DOCUMENT_TEXT.append(tokens) # for ranking
                ONLY_DOCUMENTS.append(document) # for retrieval reconstruction

        return {}

    def gen_fixed_query_dataset():
        for i in range(len(DOCUMENT_TEXT)):
            yield {"text": DOCUMENT_TEXT[i], "label": -1}

    flat = msmarco.flatten()
    filtered = flat.filter(
        lambda row: row["query"] == RANDOM_QUERY
    )
    for i in range(len(filtered["passages.passage_text"][0])):
        if filtered["passages.is_selected"][0][i] == 1:
            MATCHED_DOCUMENT = filtered["passages.passage_text"][0][i]
            break

    # QUERY_INDEX = -1
    # filtered = flat.filter(
    #     lambda row, idx: ,
    #     with_indices=True
    # )

    flat.map(
        process_batch_text,
        batched=True
    )
    fixed_query_dataset = Dataset.from_generator(gen_fixed_query_dataset)

    embedded_fixed_query_dataset = fixed_query_dataset.map(
        EMBEDDING[MODEL], 
        batched=True
    )
    text = np.array(embedded_fixed_query_dataset["text"], dtype="int32")
    label = np.array(embedded_fixed_query_dataset["label"], dtype="int32")

    model.eval()
    y_hat_all = []
    for num_batch in range(len(text) // training_config.batch_size + 1):

        log_ids = text[
            num_batch * training_config.batch_size 
            : num_batch * training_config.batch_size + training_config.batch_size, 
            : training_config.tokenizer_max_length
        ]
        targets = label[
            num_batch * training_config.batch_size 
            : num_batch * training_config.batch_size + training_config.batch_size
        ]
        log_ids = torch.LongTensor(log_ids).cuda(0, non_blocking=True)
        targets = torch.FloatTensor(targets).cuda(0, non_blocking=True)

        bz_loss3, y_hat3 = model(log_ids, targets)
        y_hat_all += y_hat3.to("cpu").detach().numpy().tolist()

    model.train()

    # sort by ranking score
    enum_scores = []
    for (idx, score) in enumerate(y_hat_all):
        enum_scores.append((idx, score))
    enum_scores = sorted(enum_scores, key=lambda x: x[1], reverse=True)
    top_k = enum_scores[:TOP_K]

    # reconstruct top k documents
    print(f"[RETRIEVAL] query: {RANDOM_QUERY}")
    print(f"[RETRIEVAL] document to retrieve: {MATCHED_DOCUMENT}")
    # inv_vocab = {v: k for k, v in VOCAB.items()}
    for (idx, score) in top_k:
        doc = ONLY_DOCUMENTS[idx]
        print(f"[RETRIEVAL] score: {score:.5}, document: '{doc}'")
print(" - done")
