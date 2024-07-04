from datasets import load_dataset, disable_caching
from transformers import BertConfig
from preprocessing import PreprocessingModule
from index_embedding import IndexEmbeddingModule
from training import TrainingModule


if __name__ == '__main__':
    CONFIG_SAVE_PATH = "config/retrieval_system/"
    pipeline_config = BertConfig.from_json_file(CONFIG_SAVE_PATH+"pipeline_config.json")
    training_config = BertConfig.from_json_file(CONFIG_SAVE_PATH+f"{pipeline_config.model}-training_config.json")
    model_config = BertConfig.from_json_file(CONFIG_SAVE_PATH+f"{pipeline_config.model}-model_config.json")

    PERFORM_RETRIEVAL = True # FF only


    # dataset loading
    disable_caching()  # needed for vocab generation!!!
    msmarco = load_dataset(
        "microsoft/ms_marco", "v2.1", split="train", verification_mode="no_checks"
    ).flatten()


    # tokenization + vocab creation
    PM = PreprocessingModule(
        training_config,
        model_config,
        pipeline_config
    )
    preprocessed_dataset, vocab = PM.execute(msmarco)

    # update model params
    model_config.num_embeddings = len(vocab)                                                                    
    model_config.vocab_size = len(vocab)

    # index embedding + padding
    IEM = IndexEmbeddingModule(
        training_config,
        model_config,
        pipeline_config
    )
    training_args = IEM.execute(preprocessed_dataset, vocab) # FF: tuple(t, l), SDE: triple(q, d, l)

    # training (+ evaluation)
    TM = TrainingModule(
        training_config,
        model_config,
        pipeline_config
    )
    TM.execute(training_args)


    # print("retrieving for a random query from all documents...")
    # if PERFORM_RETRIEVAL and (MODEL == "FF"):
    #     RANDOM_QUERY = msmarco["query"][np.random.randint(0, len(msmarco))]
    #     MATCHED_DOCUMENT = "NO_DOCUMENT_MATCHED_THE_QUERY"
    #     ONLY_DOCUMENTS = [] # for lookup of the retrieved documents' text
    #     DOCUMENT_TEXT = [] # global for constructing the embeddings
    #     TOP_K = 10 # num of retrieved docs

    #     def process_batch_text(batch):
    #         for i in range(len(batch)):
    #             for document in batch["passages.passage_text"][i]:
    #                 tokens = preprocess(" ".join([RANDOM_QUERY, document]))
    #                 DOCUMENT_TEXT.append(tokens) # for ranking
    #                 ONLY_DOCUMENTS.append(document) # for retrieval reconstruction

    #         return {}

    #     def gen_fixed_query_dataset():
    #         for i in range(len(DOCUMENT_TEXT)):
    #             yield {"text": DOCUMENT_TEXT[i], "label": -1}

    #     flat = msmarco.flatten()
    #     filtered = flat.filter(
    #         lambda row: row["query"] == RANDOM_QUERY
    #     )
    #     for i in range(len(filtered["passages.passage_text"][0])):
    #         if filtered["passages.is_selected"][0][i] == 1:
    #             MATCHED_DOCUMENT = filtered["passages.passage_text"][0][i]
    #             break

    #     # QUERY_INDEX = -1
    #     # filtered = flat.filter(
    #     #     lambda row, idx: ,
    #     #     with_indices=True
    #     # )

    #     flat.map(
    #         process_batch_text,
    #         batched=True
    #     )
    #     fixed_query_dataset = Dataset.from_generator(gen_fixed_query_dataset)

    #     embedded_fixed_query_dataset = fixed_query_dataset.map(
    #         EMBEDDING[MODEL], 
    #         batched=True
    #     )
    #     text = np.array(embedded_fixed_query_dataset["text"], dtype="int32")
    #     label = np.array(embedded_fixed_query_dataset["label"], dtype="int32")

    #     model.eval()
    #     y_hat_all = []
    #     for num_batch in range(len(text) // training_config.batch_size + 1):

    #         log_ids = text[
    #             num_batch * training_config.batch_size 
    #             : num_batch * training_config.batch_size + training_config.batch_size, 
    #             : training_config.tokenizer_max_length
    #         ]
    #         targets = label[
    #             num_batch * training_config.batch_size 
    #             : num_batch * training_config.batch_size + training_config.batch_size
    #         ]
    #         log_ids = torch.LongTensor(log_ids).cuda(0, non_blocking=True)
    #         targets = torch.FloatTensor(targets).cuda(0, non_blocking=True)

    #         bz_loss3, y_hat3 = model(log_ids, targets)
    #         y_hat_all += y_hat3.to("cpu").detach().numpy().tolist()

    #     model.train()

    #     # sort by ranking score
    #     enum_scores = []
    #     for (idx, score) in enumerate(y_hat_all):
    #         enum_scores.append((idx, score))
    #     enum_scores = sorted(enum_scores, key=lambda x: x[1], reverse=True)
    #     top_k = enum_scores[:TOP_K]

    #     # reconstruct top k documents
    #     print(f"[RETRIEVAL] query: {RANDOM_QUERY}")
    #     print(f"[RETRIEVAL] document to retrieve: {MATCHED_DOCUMENT}")
    #     # inv_vocab = {v: k for k, v in VOCAB.items()}
    #     for (idx, score) in top_k:
    #         doc = ONLY_DOCUMENTS[idx]
    #         print(f"[RETRIEVAL] score: {score:.5}, document: '{doc}'")
    # print(" - done")
