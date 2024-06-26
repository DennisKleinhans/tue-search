from datasets import Dataset, load_dataset, disable_caching, load_from_disk
from nltk.tokenize import wordpunct_tokenize
import numpy as np
import torch
from fastformer import Model
from scipy.stats import kendalltau, pearsonr
import json


LOAD_DATASET_FROM_DISK = True
DATASET_SAVE_PATH = "data/retrieval_system/"


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

training_config = BertConfig.from_dict(
    {
        "learning_rate": 1e-2, # 1e-3
        "num_epochs": 1, # 2
        "batch_size": 128,# has to be large enough to prevent constant targets
        "tokenizer_max_length": EMBEDDING_SIZE
    }
)
model_config = BertConfig.from_dict(
    {
        "num_embeddings": len(VOCAB),
        "num_labels": 1,
        "hidden_size": 256, # 256
        "hidden_dropout_prob": 0.2, # 0.2
        "num_hidden_layers": 2, # 2
        "hidden_act": "relu", # "gelu"
        "num_attention_heads": 16, # 16
        "intermediate_size": 256, # 256
        "max_position_embeddings": EMBEDDING_SIZE,
        "type_vocab_size": 2,
        "vocab_size": len(VOCAB),
        "layer_norm_eps": 1e-12,
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
    corr_coeff = 0.0
    for cnt in range(len(train_idx) // training_config.batch_size):

        log_ids = text[train_idx][
            cnt * training_config.batch_size
            : cnt * training_config.batch_size + training_config.batch_size,
            : training_config.tokenizer_max_length,
        ]
        targets = label[train_idx][
            cnt * training_config.batch_size 
            : cnt * training_config.batch_size + training_config.batch_size
        ]

        log_ids = torch.LongTensor(log_ids).cuda(0, non_blocking=True)
        targets = torch.FloatTensor(targets).cuda(0, non_blocking=True)
        bz_loss, y_hat = model(log_ids, targets)
        loss += bz_loss.data.float()
        this_coeff = pearsonr(
            targets.to("cpu").detach().numpy().tolist(),
            y_hat.to("cpu").detach().numpy().tolist(),
        ).statistic
        if np.isnan(this_coeff):
            print(f"targets: {targets}")
            print(f"y_hat: {y_hat}")
            print("ERROR: targets is constant, can't compute pcc")
            print(" - setting pcc to 0 for this batch")
            this_coeff = 0
        corr_coeff += this_coeff

        unified_loss = bz_loss
        optimizer.zero_grad()
        unified_loss.backward()
        optimizer.step()

        if cnt % 100 == 0:
            print(
                "[TRAIN SET] Epoch: {}, Samples: 0-{}, train_loss: {:.5f}, pcc: {:.5f}".format(
                    i+1,
                    training_config.batch_size + (cnt * training_config.batch_size), 
                    loss.data / (cnt + 1), # mean over all batches processed so far
                    corr_coeff / (cnt + 1) # mean over all batches processed so far
                )
            )
    model.eval()
    y_hat_all = []
    for cnt in range(len(test_idx) // training_config.batch_size + 1):

        log_ids = text[test_idx][
            cnt * training_config.batch_size 
            : cnt * training_config.batch_size + training_config.batch_size, 
            : training_config.tokenizer_max_length
        ]
        targets = label[test_idx][
            cnt * training_config.batch_size 
            : cnt * training_config.batch_size + training_config.batch_size
        ]
        log_ids = torch.LongTensor(log_ids).cuda(0, non_blocking=True)
        targets = torch.FloatTensor(targets).cuda(0, non_blocking=True)

        bz_loss2, y_hat2 = model(log_ids, targets)
        y_hat_all += y_hat2.to("cpu").detach().numpy().tolist()

    y_true = label[test_idx]
    print("[TEST SET] Pearson correlation coefficient after epoch {}: {:.5f}\n".format(
        i+1,
        pearsonr(y_true, y_hat_all).statistic
    ))

    model.train()
print(" - done")


print("retrieving for a random query from all documents...")
RANDOM_QUERY = msmarco["query"][np.random.randint(0, len(msmarco))]
ONLY_DOCUMENTS = []
DOCUMENT_TEXT = []
TOP_K = 10

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

msmarco.flatten().map(
    process_batch_text,
    batched=True
)
fixed_query_dataset = Dataset.from_generator(gen_fixed_query_dataset)

embedded_fixed_query_dataset = fixed_query_dataset.map(
    embed_batch, 
    batched=True
)
text = np.array(embedded_fixed_query_dataset["text"], dtype="int32")
label = np.array(embedded_fixed_query_dataset["label"], dtype="int32")

model.eval()
y_hat_all = []
for cnt in range(len(test_idx) // training_config.batch_size + 1):

    log_ids = text[
        cnt * training_config.batch_size 
        : cnt * training_config.batch_size + training_config.batch_size, 
        : training_config.tokenizer_max_length
    ]
    targets = label[
        cnt * training_config.batch_size 
        : cnt * training_config.batch_size + training_config.batch_size
    ]
    log_ids = torch.LongTensor(log_ids).cuda(0, non_blocking=True)
    targets = torch.FloatTensor(targets).cuda(0, non_blocking=True)

    bz_loss3, y_hat3 = model(log_ids, targets)
    y_hat_all += y_hat3.to("cpu").detach().numpy().tolist()

model.train()

# sort by ranking score
# print(y_hat_all)
# print(y_hat3)
enum_scores = []
for (idx, score) in enumerate(y_hat_all):
    enum_scores.append((idx, score))
enum_scores = sorted(enum_scores, key=lambda x: x[1])
top_k = enum_scores[:TOP_K]

# reconstruct top k documents
print(f"query: {RANDOM_QUERY}")
# inv_vocab = {v: k for k, v in VOCAB.items()}
for (idx, score) in top_k:
    doc = ONLY_DOCUMENTS[idx]
    print(f"score: {score:.5}, document: '{doc}'")


print(" - done")
