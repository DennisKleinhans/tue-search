import numpy as np
from dataset_utils import prepare_msmarco_dataset
from Predictors import RandomPredictor, QueryLikelihoodPredictor, TFIDFPredictor

# recall of the top k documents
def topk_recall(y_hat, y_test, k=1):
    num_examples = float(len(y_hat))
    num_correct = 0
    for pred, test_idx in zip(y_hat, y_test):
        if test_idx in pred[:k]:
            num_correct += 1
    return num_correct / num_examples


def evaluate_topk_recall(preds, test, max_k=10):
    for n in range(1, max_k+1, 2):
        print("Recall @ ({}, {}): {:g}".format(n, max_k, topk_recall(preds, test, n)))
    print()


if __name__ == "__main__":
    msmarco_train, msmarco_test = prepare_msmarco_dataset(
        # drop_idx=10000
    )

    # print(msmacro_test["query"][0])
    # print(msmacro_test["passages"][0])

    # evaluate Random predictor
    randomPred = RandomPredictor()
    print("Random Predictor Performance")
    randomPred.evaluate(msmarco_test, evaluate_topk_recall)

    # evaluate query likelihood predictor
    QLPred = QueryLikelihoodPredictor()
    print("Query Likelihood Predictor Performance")
    QLPred.evaluate(msmarco_test, evaluate_topk_recall)

    # evaluate tfidf predictor
    TFIDFPred = TFIDFPredictor()
    print("TFIDF Predictor Performance")
    TFIDFPred.train(msmarco_train)
    TFIDFPred.evaluate(msmarco_test, evaluate_topk_recall)