import sys
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


if __name__ == "__main__":
    sys.path.insert(0, f"{os.getcwd()}")
    from src.retrieval_system.logistic_regression.retrieval_interface import RetrievalSystemInterface
    from src.retrieval_system.logistic_regression.preprocessing import preprocess

    RSI = RetrievalSystemInterface()
    RSI.train_retrieval_system()
    