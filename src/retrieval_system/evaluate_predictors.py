import numpy as np
from dataset_utils import prepare_msmacro_dataset
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
    msmacro_train, msmacro_test = prepare_msmacro_dataset(
        # drop_idx=10000
    )

    # print(msmacro_test["query"][0])
    # print(msmacro_test["passages"][0])

    # evaluate Random predictor
    randomPred = RandomPredictor()
    print("Random Predictor Performance")
    randomPred.evaluate(msmacro_test, evaluate_topk_recall)

    # evaluate query likelihood predictor
    QLPred = QueryLikelihoodPredictor()
    print("Query Likelihood Predictor Performance")
    QLPred.evaluate(msmacro_test, evaluate_topk_recall)

    # evaluate tfidf predictor
    TFIDFPred = TFIDFPredictor()
    print("TFIDF Predictor Performance")
    TFIDFPred.train(msmacro_train)
    TFIDFPred.evaluate(msmacro_test, evaluate_topk_recall)