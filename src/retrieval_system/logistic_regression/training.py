from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_from_disk
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk import ngrams, edit_distance
from sklearn.metrics import f1_score, precision_score, recall_score
import dill
from sklearn.metrics import ndcg_score
from nltk.metrics import ConfusionMatrix
import sys
import os

sys.path.insert(0, f"{os.getcwd()}")
from src.retrieval_system.logistic_regression.module_classes import ProcessingModule


def get_log_prob(probs):
    result = 0
    for p in probs:
        result += np.log2([p])[0]
    return np.exp([result])[0]


def get_ngram_prob(ngram, lm_tuple):
    ngram_lm, unseen_prob = lm_tuple
    result = 0.0
    try:
        result = ngram_lm[ngram]
    except KeyError:
        result = unseen_prob
    return result


def get_ngram_lm(corpus_tokens, pad_token, n, alpha=0.5):
    """returns tuple: (dict[ngram] = prob of ngram, prob for unseen ngram)"""

    def laplace_smoothing(count, dim, trials, alpha=alpha):
        theta = (count + alpha) / (trials + alpha * dim)
        return trials * theta

    lm = {}
    _ngrams = list(ngrams(corpus_tokens, n))
    for ngram in _ngrams:
        if ngram.count(pad_token) < n:
            try:
                lm[ngram] += 1
            except KeyError:
                lm[ngram] = 1

    if n > 1:
        _smaller_ngrams = list(ngrams(corpus_tokens, n - 1))
        num_smaller_ngram = {}
        for s_ngram in _smaller_ngrams:
            try:
                num_smaller_ngram[s_ngram] += 1
            except KeyError:
                num_smaller_ngram[s_ngram] = 1

    zero_occurrence_prob = 1e-10  # is small
    for ngram in lm:
        if n > 1:  # MLE with relative frequency
            # given ngram x_1...x_n
            # returns P(x_n|x_1:n-1) = count(x_1:n)/count(x_1:n-1)
            s_ngram = ngram[: len(ngram) - 1]
            num_s_ngram = num_smaller_ngram[s_ngram]
            lm[ngram] = lm[ngram] / num_s_ngram
        else:  # unigrams
            # given unigram x
            # returns count(x)/len(all unigrams)
            lm[ngram] = lm[ngram] / len(_ngrams)
    return (lm, zero_occurrence_prob)


def weighted_ngram_matching(seq1, seq2, ngram_lms, n):
    # counts the number of matching ngrams weighted by the inverse prob of that ngram
    # such that rare matching ngrams result in a large coefficient (in log space)
    ngrams1 = set(ngrams(seq1, n))
    ngrams2 = set(ngrams(seq2, n))
    inv_ngram_probs = []
    for ngram in ngrams1:
        if ngram in ngrams2:
            inv_ngram_probs.append(1 - get_ngram_prob(ngram, ngram_lms[n - 1]))
    return get_log_prob(inv_ngram_probs)
    # return np.sum(inv_ngram_probs)/len(ngrams1)


def count_matching_ngrams(qry, doc, n):
    ngrams1 = set(ngrams(qry, n))
    ngrams2 = set(ngrams(doc, n))
    count = 0
    for ngram in ngrams1:
        if ngram in ngrams2:
            count += 1
    return count / len(ngrams1)


def normalized_levenshtein_distance(query, document, subst_cost):
    norm = max(len(query), len(document))
    return edit_distance(query, document, substitution_cost=subst_cost) / (
        norm * subst_cost
    )


def get_features(qry, doc, qry_embeds, doc_embeds, vectorizer, ngram_lms, feature_set):
    query = " ".join(qry)
    document = " ".join(doc)

    tfidf = vectorizer.fit_transform([query, document])

    result = []

    if feature_set == 1:  #
        result.append(cosine_similarity(tfidf[0], tfidf[1])[0][0])

    elif feature_set == 2:  #
        result.append(cosine_similarity(tfidf[0], tfidf[1])[0][0])
        result.append(weighted_ngram_matching(query, document, ngram_lms, n=1))

    elif feature_set == 3:  #
        result.append(cosine_similarity(tfidf[0], tfidf[1])[0][0])
        result.append(weighted_ngram_matching(query, document, ngram_lms, n=1))
        result.append(weighted_ngram_matching(query, document, ngram_lms, n=2))

    elif feature_set == 4:  #
        result.append(cosine_similarity(tfidf[0], tfidf[1])[0][0])
        result.append(weighted_ngram_matching(query, document, ngram_lms, n=1))
        result.append(weighted_ngram_matching(query, document, ngram_lms, n=2))
        result.append(edit_distance(qry, doc, substitution_cost=1))

    elif feature_set == 5:  #
        result.append(cosine_similarity(tfidf[0], tfidf[1])[0][0])
        result.append(weighted_ngram_matching(query, document, ngram_lms, n=1))
        result.append(weighted_ngram_matching(query, document, ngram_lms, n=2))
        result.append(edit_distance(qry, doc, substitution_cost=2))

    elif feature_set == 6:  #
        result.append(cosine_similarity(tfidf[0], tfidf[1])[0][0])
        result.append(weighted_ngram_matching(query, document, ngram_lms, n=2))
        result.append(weighted_ngram_matching(query, document, ngram_lms, n=3))
        result.append(edit_distance(qry, doc, substitution_cost=1))

    elif feature_set == 7:  #
        result.append(cosine_similarity(tfidf[0], tfidf[1])[0][0])
        result.append(weighted_ngram_matching(query, document, ngram_lms, n=2))
        result.append(weighted_ngram_matching(query, document, ngram_lms, n=3))
        result.append(edit_distance(qry, doc, substitution_cost=2))

    else:
        raise ValueError(f"feature set {feature_set} is not implemented.")

    return result


def process_batch(batch, vectorizer, ngram_lms, feature_set, batched):
    if batched:
        f = []
        t = []
        for i in range(len(batch)):
            f.append(
                get_features(
                    batch["query"][i],
                    batch["document"][i],
                    batch["query_embeds"][i],
                    batch["document_embeds"][i],
                    vectorizer,
                    ngram_lms,
                    feature_set,
                )
            )
            t.append(batch["target"][i])
        return {"features": f, "targets": t}
    else:
        return {
            "features": get_features(
                batch["query"],
                batch["document"],
                batch["query_embeds"],
                batch["document_embeds"],
                vectorizer,
                ngram_lms,
                feature_set,
            ),
            "targets": batch["target"],
        }


class TrainingModuleV2(ProcessingModule):
    def __init__(self, train_config, model_config, pipeline_config) -> None:
        super().__init__(train_config, model_config)
        self.pipeline_config = pipeline_config

        self.model = None
        if pipeline_config.model == "LR":
            self.model = LogisticRegression(
                solver="lbfgs", penalty="l2", class_weight="balanced"
            )
        elif pipeline_config.model == "SVM":
            self.model = SVC(kernel=model_config.kernel, probability=True)
        else:
            raise NotImplementedError(
                f"The '{pipeline_config.model}' model is not implemented."
            )

        self.vectorizer = TfidfVectorizer(analyzer="word", stop_words="english")
        self.model_save_path = (
            self.pipeline_config.dataset_save_path
            + f"/{self.pipeline_config.model}-model-fs{self.train_config.feature_set}.pickle"
        )
        self.ngram_lms_save_path = (
            self.pipeline_config.dataset_save_path + "ngram_lms.pickle"
        )

        # filled in execute()
        self.embedding_map = None
        self.ngram_n = range(1, 4)
        self.ngram_lms = []

    def save_model(self):
        with open(self.model_save_path, "wb") as f:
            dill.settings["recurse"] = True
            dill.dump(self.model, f, protocol=dill.HIGHEST_PROTOCOL)

    def load_model(self):
        with open(self.model_save_path, "rb") as f:
            value = dill.load(f)
            self.model = value

    def execute(self, preprocessed_dataset):
        if (
            not self.pipeline_config.load_mapped_features_from_disk
            or not self.pipeline_config.load_dataset_from_disk
        ):
            print("creating ngram models...")
            recompute = (
                self.pipeline_config.recompute_ngram_lms
                or not self.pipeline_config.load_dataset_from_disk
            )
            try:
                with open(self.ngram_lms_save_path, "rb") as f:
                    self.ngram_lms = dill.load(f)
            except FileNotFoundError:
                recompute = True

            if recompute:
                corpus = []
                for row in preprocessed_dataset:
                    for token in row["query"]:
                        corpus.append(token)
                    for token in row["document"]:
                        corpus.append(token)
                for n in self.ngram_n:
                    print(f" creating {n}-gram LM...")
                    self.ngram_lms.append(
                        get_ngram_lm(corpus, self.train_config.pad_token, n)
                    )
                with open(self.ngram_lms_save_path, "wb") as f:
                    dill.settings["recurse"] = True
                    dill.dump(self.ngram_lms, f, protocol=dill.HIGHEST_PROTOCOL)
            print(" - done")

        print("mapping features...")
        dataset_savename = f"{self.pipeline_config.dataset_save_path}mapped_features_bs{self.train_config.batch_size}_embed-{self.pipeline_config.embedding_type}"
        if self.pipeline_config.embedding_type == "glove":
            dataset_savename += "-".join(
                self.pipeline_config.glove_file.lstrip("glove")
                .rstrip(".txt")
                .split(".")
            )
        if self.train_config.batch_padding:
            dataset_savename += "_padded"
        dataset_savename += f"_tml{self.train_config.tokenizer_max_length}_fs{self.train_config.feature_set}"

        if self.pipeline_config.load_mapped_features_from_disk:
            mapped_features = load_from_disk(dataset_savename)
            print(" - loaded from disk")
        else:
            _batched = False
            mapped_features = preprocessed_dataset.map(
                lambda batch: process_batch(
                    batch,
                    self.vectorizer,
                    self.ngram_lms,
                    self.train_config.feature_set,
                    batched=_batched,
                ),
                batched=_batched,
                remove_columns=preprocessed_dataset.column_names,
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
        train_features = np.asarray([features[i] for i in train_idx], dtype=float)
        if self.pipeline_config.load_model_weights_from_disk:
            self.load_model()
        else:
            self.model.fit(train_features, train_targets)
            self.save_model()
        print(" - done")

        print("evaluating model...")
        eval_targets = np.asarray([targets[i] for i in test_idx], dtype=int)
        eval_features = np.asarray([features[i] for i in test_idx], dtype=float)

        y_hat = [
            np.argmax(scores) for scores in self.model.predict_proba(eval_features)
        ]
        # print(y_hat)

        # pretty evaluation metrics
        true = [[1, 0] if x == 0 else [0, 1] for x in eval_targets]
        hat = self.model.predict_proba(eval_features)

        print(
            f"\n{self.pipeline_config.model} - Feature Set {self.train_config.feature_set}"
        )
        print(f"Precision: {precision_score(eval_targets, y_hat)}")
        print(f"Recall: {recall_score(eval_targets, y_hat)}")
        print(f"macro-avg. F1: {f1_score(eval_targets, y_hat, average='macro')}")
        print(f"NDCG@10: {ndcg_score(true, hat, k=10)}\n")
        print(
            ConfusionMatrix(eval_targets, y_hat).pretty_format(
                show_percents=True,
                values_in_chart=True,
                truncate=15,
                sort_by_count=True,
            )
        )
        print(" - done")

    def retrieve(self, search_query, search_documents, doc_ids):
        # load model weights
        try:
            self.load_model()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model weights file {self.model_save_path} not found. Train the model before attempting search retrieval."
            )

        # load and prepare ngram LMs
        try:
            with open(self.ngram_lms_save_path, "rb") as f:
                self.ngram_lms = dill.load(f)
                f.close()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Ngram LMs file {self.ngram_lms_save_path} not found. Train the model before attempting search retrieval."
            )

        # glove embedding catch
        # implementing this is currently not planned bc glove embedding does not improve performance much anyway
        if self.pipeline_config.embedding_type != "none":
            raise NotImplementedError(
                f"Embedding type '{self.pipeline_config.embedding_type}' is not supported for search retrieval!"
            )

        # actual retrieval :)
        features = [
            get_features(
                search_query,
                doc_tokens,
                [],
                [],
                self.vectorizer,
                self.ngram_lms,
                self.train_config.feature_set,
            )
            for doc_tokens in search_documents
        ]

        probs = [scores[1] for scores in self.model.predict_proba(features)]

        return_tuples = [
            (search_documents[i], probs[i], doc_ids[i])
            for i in range(len(search_documents))
        ]
        return_tuples.sort(key=lambda x: x[1], reverse=True)

        return return_tuples
