import pandas as pd
import numpy as np
import re
import nltk
import contractions
from nltk.corpus import stopwords
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.utils import to_categorical

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_review(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = contractions.fix(text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

def remove_stopwords(text):
    return " ".join(w for w in text.split() if w not in stop_words)

def load_and_prepare_data(nrows, max_tokens, max_length):
    df = pd.read_csv(
        "../Data/train.csv",
        header=None,
        names=["label", "title", "review"],
        nrows=nrows
    )

    df["clean"] = (
        df["review"]
        .astype(str)
        .apply(clean_review)
        .apply(remove_stopwords)
    )

    labels = df["label"].values - 1
    num_classes = len(np.unique(labels))
    y_train = to_categorical(labels, num_classes)

    vectorizer = TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=max_length
    )

    vectorizer.adapt(df["clean"].values)
    X_train = vectorizer(df["clean"].values)

    test_df = pd.read_csv(
        "../Data/test.csv",
        header=None,
        names=["label", "title", "review"],
        nrows=nrows
    )

    test_df["clean"] = (
        test_df["review"]
        .astype(str)
        .apply(clean_review)
        .apply(remove_stopwords)
    )

    X_val = vectorizer(test_df["clean"].values)
    y_val = to_categorical(test_df["label"].values - 1, num_classes)

   return X_train, y_train, X_val, y_val, vectorizer

