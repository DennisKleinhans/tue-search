from abc import ABCMeta, abstractmethod
from time import time
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy.linalg import norm


class Predictor(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def train(self, dataset):
        """trains the predictor on the given dataset"""
        pass
    
    @abstractmethod
    def predict(self, query, documents):
        """for a given query, returns a list of document indices sorted by ranking score"""
        pass

    def evaluate(self, dataset, eval_metric):
        """applies the given eval_metric to a set of predictions on the given dataset"""

        def y_test_util(is_selected):
            result = None
            try:
                result = is_selected.index(1)
            except ValueError:
                pass
            return result

        mapped_dataset = dataset.map(
            lambda batch: {
                "y_hat": [self.predict(batch["query"][i], batch["passages"][i]["passage_text"]) for i in range(len(batch))],
                "y_test": [y_test_util(batch["passages"][i]["is_selected"]) for i in range(len(batch))]
            },
            remove_columns=dataset.column_names,
            batched=True
        )
        
        eval_metric(mapped_dataset["y_hat"], mapped_dataset["y_test"])


class RandomPredictor(Predictor):
    def __init__(self) -> None:
        pass

    def train(self, dataset):
        pass

    def predict(self, query, documents):
        return np.random.choice(len(documents), len(documents), replace=False)
    

class QueryLikelihoodPredictor(Predictor):
    def __init__(self) -> None:
        pass

    def train(self, dataset):
        pass

    def predict(self, query, documents):
        ranked_doc_indices = []
        # preprocessing
        documents = [nltk.tokenize.wordpunct_tokenize(d) for d in documents]
        query = nltk.tokenize.wordpunct_tokenize(query)

        # ranking using unigram lm + laplace smoothing
        for idx in range(len(documents)):
            r = 1.0 # rank
            for q in query:
                r *= self._get_token_prob_from_unigram_LM(q, documents[idx])
            ranked_doc_indices.append((idx, r))
        
        ranked_doc_indices.sort(key=lambda t: t[1])
        return [t[0] for t in ranked_doc_indices]
            

    def _get_token_prob_from_unigram_LM(self, token, tokens, alpha=0.5):
        token_freqs = {} # counts token occurences
        for t in tokens:
            try:
                token_freqs[t] += 1
            except KeyError:
                token_freqs[t] = 1

        # laplace smoothing
        for key in token_freqs:
            N = len(tokens) # document size
            d = len(token_freqs.items()) # vocab size
            token_freqs[key] = (token_freqs[key]+alpha)/(N+alpha*d)
        token_prob = 0.0

        # retrieve token prob
        try:
            token_prob = token_freqs[token]
        except KeyError:
            N = len(tokens)
            d = len(token_freqs.items())
            token_prob = alpha/(N+alpha*d)
        return token_prob
    

class TFIDFPredictor(Predictor):
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def train(self, dataset):
        # learn vocab and idf
        mapped_dataset = dataset.map(
            lambda batch: {
                "train_documents": np.append(
                    [document for i in range(len(batch)) for document_list in batch["passages"][i]["passage_text"] for document in document_list], 
                    [batch["query"][i] for i in range(len(batch))]
                )
            },
            remove_columns=dataset.column_names,
            batched=True
        )
        self.vectorizer.fit(mapped_dataset["train_documents"])

    def predict(self, query, documents):
        vector_qry = self.vectorizer.transform([query])
        vector_doc = self.vectorizer.transform(documents)

        result = np.dot(vector_doc, vector_qry.T).todense()
        result = np.asarray(result).flatten()

        return np.argsort(result, axis=0)[::-1]
