import sys
import os


if __name__ == "__main__":
    sys.path.insert(0, f"{os.getcwd()}")
    from src.retrieval_system.logistic_regression.retrieval_interface import RetrievalSystemInterface

    RSI = RetrievalSystemInterface()
    RSI.train_retrieval_system()