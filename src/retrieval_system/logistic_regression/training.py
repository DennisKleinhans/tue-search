from module_classes import ProcessingModule
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import wordpunct_tokenize
from time import time
from datasets import load_from_disk, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.metrics import ndcg_score
from nltk.metrics import ConfusionMatrix


def reciprocal_rank(true, hat):
    """"computes the rr for one batch (aka true should have only one 1)"""
    sorted_ranks = sorted(enumerate(hat), key=lambda t: t[1], reverse=True)
    ground_truth_rank = -1
    for i, x in sorted_ranks:
        try:
            if i == list(true).index(1):
                ground_truth_rank = i+1
                break
        except ValueError: # '1' not in 'true'
            break
    if ground_truth_rank != -1:
        return 1/ground_truth_rank
    else:
        return 0
    

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


def get_features(qry_tokens, doc_tokens, embed_map, vectorizer, feature_set):
    query = " ".join(qry_tokens)
    qry_embeds = get_glove_embed(qry_tokens, embed_map)
    qry_embeds_1d = np.mean(qry_embeds, axis=0)

    document = " ".join(doc_tokens)
    doc_embeds = get_glove_embed(doc_tokens, embed_map)
    doc_embeds_1d = np.mean(doc_embeds, axis=0)

    tfidf = vectorizer.fit_transform([query, document])

    # features
    tfidf_cos = cosine_similarity(tfidf[0], tfidf[1])
    embed_cos = cosine_similarity(qry_embeds_1d.reshape(1, -1), doc_embeds_1d.reshape(1, -1))

    if feature_set == 1:
        return [
            tfidf_cos,
            embed_cos
        ]
    elif feature_set == 2:
        result = []
        result.append(tfidf_cos)
        result.append(embed_cos)
        for token in embed_map:
            result.append(qry_tokens.count(token)/len(embed_map))
        return result
    else:
        raise ValueError(f"feature set {feature_set} is not implemented.")


def process_batch(batch, embed_map, vectorizer, feature_set, batched=True):
    if batched:
        f = []
        t = []
        for i in range(len(batch)):
            f.append(get_features(batch["query"][i], batch["document"][i], embed_map, vectorizer, feature_set))
            t.append(batch["target"][i])
        return {"features": f, "targets": t}
    else:
        return {"features": get_features(batch["query"], batch["document"], embed_map, vectorizer, feature_set), "targets": batch["target"]}


class TrainingModuleV2(ProcessingModule):
    def __init__(self, train_config, model_config, pipeline_config) -> None:
        super().__init__(train_config, model_config)
        self.pipeline_config = pipeline_config

        self.model = LogisticRegression(solver="lbfgs", penalty="l2", class_weight="balanced")
        self.vectorizer = TfidfVectorizer(analyzer='word', stop_words="english")

        # filled in execute()
        self.embedding_map = None


    def execute(self, preprocessed_dataset, embed_map):
        self.embedding_map = embed_map
        
        print("mapping features...")
        dataset_savename = f"{self.pipeline_config.dataset_save_path}{self.pipeline_config.model}-mapped_features_bs{self.train_config.batch_size}_embed-{self.pipeline_config.embedding_type}"
        if self.pipeline_config.embedding_type == "glove":
            dataset_savename += "-".join(self.pipeline_config.glove_file.lstrip("glove").rstrip(".txt").split("."))
        if self.train_config.batch_padding:
            dataset_savename += "_padded"
        dataset_savename += f"_tml{self.train_config.tokenizer_max_length}_fs{self.train_config.feature_set}"

        if self.pipeline_config.load_dataset_from_disk:
            mapped_features = load_from_disk(dataset_savename)
            print(" - loaded from disk")
        else:
            mapped_features = preprocessed_dataset.map(
                lambda batch: process_batch(batch, self.embedding_map, self.vectorizer, self.train_config.feature_set, batched=False),
                batched=False,
                remove_columns=preprocessed_dataset.column_names
            )
            mapped_features.save_to_disk(dataset_savename)
            print(" - done")


        features = mapped_features["features"]
        targets = mapped_features["targets"]

        
        print("creating split indices...")
        index = np.arange(len(features), dtype=int)
        breakpoint = int(np.ceil(self.train_config.train_split_size * len(index)))
        train_idx = index[:breakpoint]
        if not self.train_config.batch_padding:
            np.random.shuffle(train_idx)
        test_idx = index[breakpoint:]
        print(" - done")


        print("training model...")
        train_targets = np.asarray([targets[i] for i in train_idx], dtype=int)
        train_features = np.asarray([features[i] for i in train_idx], dtype=float).reshape(len(train_targets), 2)
        # print(train_features.shape)
        # print(train_targets.shape)
        self.model.fit(train_features, train_targets)
        print(" - done")


        print("evaluating model...")
        eval_targets = np.asarray([targets[i] for i in test_idx], dtype=int)
        eval_features = np.asarray([features[i] for i in test_idx], dtype=float).reshape(len(eval_targets), 2)

        y_hat = [np.argmax(scores) for scores in self.model.predict_proba(eval_features)]
        # print(y_hat)

        wrong_unrelated = []
        right_unrelated = []
        wrong_related = []
        right_related = []
        start = time()
        for i in range(len(y_hat)):
            interval = time() - start
            if (i + 1) % 10 == 0 and interval > 0:
                print(" Reading {:>10,d} rows, {:>5.2f} rows/sec ".format(i+1,(i+1)/interval), end="\r")

            if eval_targets[i] == 1:
                if y_hat[i] > .5:
                    right_related.append(y_hat[i])
                else:
                    wrong_unrelated.append(y_hat[i])
            else:
                if y_hat[i] > .5:
                    wrong_related.append(y_hat[i])
                else:
                    right_unrelated.append(y_hat[i])
        print()
        # print("right related: ", len(right_related),
        #       "wrong related: ", len(wrong_related),
        #       "right unrelated: ", len(right_unrelated),
        #       "wrong unrelated: ", len(wrong_unrelated), )
        
        # print(self.model.score(eval_features, eval_targets))
        # print(eval_targets[:10])
        # print(y_hat[:10])


        # pretty evaluation metrics
        true = [[1,0] if x == 0 else [0,1] for x in eval_targets]
        hat = self.model.predict_proba(eval_features)
        print(f"NDCG@10: {ndcg_score(true, hat, k=10)}\n")
        print(ConfusionMatrix(eval_targets, y_hat).pretty_format(
            show_percents=True, 
            values_in_chart=True,
            truncate=15,
            sort_by_count=True
        ), end="\n\n")
        print(" - done")
