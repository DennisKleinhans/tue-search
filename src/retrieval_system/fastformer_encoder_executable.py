from datasets import Dataset, load_dataset, disable_caching, load_from_disk
from nltk.tokenize import wordpunct_tokenize
import numpy as np
import torch
from fastformer import Model
from scipy.stats import kendalltau, pearsonr
from sklearn.metrics import ndcg_score
import json


LOAD_DATASET_FROM_DISK = True
DATASET_SAVE_PATH = "data/retrieval_system/"

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
)


# tokenization + vocab creation
print("tokenizing dataset...")
VOCAB = {
    "<PAD>": 0,
    "<UNK>": 1
}
TOKENIZED_TEXT = []
LABELS = []

def get_vocab_encoding(token):
    result = VOCAB["<UNK>"]
    try:
        result = VOCAB[token]
    except KeyError:
        pass
    return result

def process_batch(batch):
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

def gen_preprocessed_dataset():
    for i in range(len(TOKENIZED_TEXT)):
        yield {"text": TOKENIZED_TEXT[i], "label": LABELS[i]}

if LOAD_DATASET_FROM_DISK:
    preprocessed_dataset = load_from_disk(DATASET_SAVE_PATH+"preprocessed_dataset")
    with open(DATASET_SAVE_PATH+"vocab.json", "r", encoding="utf-8") as fs:
        VOCAB = json.load(fs)
        fs.close()
else:
    msmarco.flatten().map(
        process_batch,
        batched=True
    )
    preprocessed_dataset = Dataset.from_generator(gen_preprocessed_dataset)

    # print(len(msmarco.flatten()))
    # print(len(preprocessed_dataset))
    # exit()

    preprocessed_dataset.save_to_disk(DATASET_SAVE_PATH+"preprocessed_dataset")
    with open(DATASET_SAVE_PATH+"vocab.json", "w", encoding="utf-8") as fs:
        json.dump(VOCAB, fs, indent=4)
        fs.close()
print(" - done")


# index embedding + padding
print("creating text embeddings...")
EMBEDDING_SIZE = 128 # 256

def embed_batch(batch):
    embeddings = [
        [get_vocab_encoding(token) for token in document][:EMBEDDING_SIZE]  # truncation
        for document in batch["text"]
    ]
    embeddings = [
        embed + [0] * (EMBEDDING_SIZE - len(embed)) for embed in embeddings  # padding
    ]
    return {"text": embeddings, "label": batch["label"]}

if LOAD_DATASET_FROM_DISK:
    embedded_dataset = load_from_disk(DATASET_SAVE_PATH+"embedded_dataset")
else:
    embedded_dataset = preprocessed_dataset.map(
        embed_batch, 
        batched=True
    )
    embedded_dataset.save_to_disk(DATASET_SAVE_PATH+"embedded_dataset")

# size = 0
# for row in embedded_dataset:
#     if size < len(row["text"]):
#         size = len(row["text"])
# print(size)
print(" - done")


print("converting dataset to numpy int32...")
text = np.array(embedded_dataset["text"], dtype="int32")
label = np.array(embedded_dataset["label"], dtype="int32")
print(" - done")


print("creating split indices...")
TRAIN_SET_SIZE = 0.8
index = np.arange(len(label))
breakpoint = int(np.ceil(TRAIN_SET_SIZE * len(text)))
train_idx = index[:breakpoint]
np.random.shuffle(train_idx)
test_idx = index[breakpoint:]
print(" - done")


print("training...")
from transformers import BertConfig
FFN_LAYER_SIZE = 256

training_config = BertConfig.from_dict(
    {
        "learning_rate": 1e-3, # 1e-3
        "num_epochs": 3, # 2
        "batch_size": 128,# has to be large enough to prevent constant targets
        "tokenizer_max_length": EMBEDDING_SIZE
    }
)
model_config = BertConfig.from_dict(
    {
        "num_embeddings": len(VOCAB),
        "num_labels": 1,
        "hidden_size": FFN_LAYER_SIZE, # 256
        "hidden_dropout_prob": 0.2, # 0.2
        "num_hidden_layers": 1, # 2
        "hidden_act": "gelu", # "gelu"
        "loss_fn": "mse",
        "num_attention_heads": 16, # 16
        "intermediate_size": FFN_LAYER_SIZE, # 256
        "max_position_embeddings": EMBEDDING_SIZE,
        "type_vocab_size": 2,
        "vocab_size": len(VOCAB),
        "layer_norm_eps": 1e-12,
        "initializer_mean": 0.0, # 0.0
        "initializer_range": 0.02, # 0.02
        "pooler_type": "weightpooler",
        "enable_fp16": "False",
    }
)

model = Model(model_config)

import torch.optim as optim
optimizer = optim.Adam(
    [{"params": model.parameters(), "lr": training_config.learning_rate}]
)

model.cuda()

for i in range(training_config.num_epochs):
    loss = 0.0
    accuary = 0.0
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

print(" - done")


print("retrieving for a random query from all documents...")
# RANDOM_QUERY = msmarco["query"][np.random.randint(0, len(msmarco))]
# MATCHED_DOCUMENT = "NO_DOCUMENT_MATCHED_THE_QUERY"
# ONLY_DOCUMENTS = [] # for lookup of the retrieved documents' text
# DOCUMENT_TEXT = [] # global for constructing the embeddings
# TOP_K = 10 # num of retrieved docs

# def process_batch_text(batch):
#     for i in range(len(batch)):
#         for document in batch["passages.passage_text"][i]:
#             tokens = preprocess(" ".join([RANDOM_QUERY, document]))
#             DOCUMENT_TEXT.append(tokens) # for ranking
#             ONLY_DOCUMENTS.append(document) # for retrieval reconstruction

#     return {}

# def gen_fixed_query_dataset():
#     for i in range(len(DOCUMENT_TEXT)):
#         yield {"text": DOCUMENT_TEXT[i], "label": -1}

# flat = msmarco.flatten()
# filtered = flat.filter(
#     lambda row: row["query"] == RANDOM_QUERY
# )
# for i in range(len(filtered["passages.passage_text"][0])):
#     if filtered["passages.is_selected"][0][i] == 1:
#         MATCHED_DOCUMENT = filtered["passages.passage_text"][0][i]
#         break

# # QUERY_INDEX = -1
# # filtered = flat.filter(
# #     lambda row, idx: ,
# #     with_indices=True
# # )

# flat.map(
#     process_batch_text,
#     batched=True
# )
# fixed_query_dataset = Dataset.from_generator(gen_fixed_query_dataset)

# embedded_fixed_query_dataset = fixed_query_dataset.map(
#     embed_batch, 
#     batched=True
# )
# text = np.array(embedded_fixed_query_dataset["text"], dtype="int32")
# label = np.array(embedded_fixed_query_dataset["label"], dtype="int32")

# model.eval()
# y_hat_all = []
# for num_batch in range(len(text) // training_config.batch_size + 1):

#     log_ids = text[
#         num_batch * training_config.batch_size 
#         : num_batch * training_config.batch_size + training_config.batch_size, 
#         : training_config.tokenizer_max_length
#     ]
#     targets = label[
#         num_batch * training_config.batch_size 
#         : num_batch * training_config.batch_size + training_config.batch_size
#     ]
#     log_ids = torch.LongTensor(log_ids).cuda(0, non_blocking=True)
#     targets = torch.FloatTensor(targets).cuda(0, non_blocking=True)

#     bz_loss3, y_hat3 = model(log_ids, targets)
#     y_hat_all += y_hat3.to("cpu").detach().numpy().tolist()

# model.train()

# # sort by ranking score
# enum_scores = []
# for (idx, score) in enumerate(y_hat_all):
#     enum_scores.append((idx, score))
# enum_scores = sorted(enum_scores, key=lambda x: x[1], reverse=True)
# top_k = enum_scores[:TOP_K]

# # reconstruct top k documents
# print(f"[RETRIEVAL] query: {RANDOM_QUERY}")
# print(f"[RETRIEVAL] document to retrieve: {MATCHED_DOCUMENT}")
# # inv_vocab = {v: k for k, v in VOCAB.items()}
# for (idx, score) in top_k:
#     doc = ONLY_DOCUMENTS[idx]
#     print(f"[RETRIEVAL] score: {score:.5}, document: '{doc}'")


print(" - done")
