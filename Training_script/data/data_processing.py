import pandas as pd
import numpy as np
import pickle
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# #nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    tokens = text.split()
    filtered = [w for w in tokens if w.lower() not in stop_words]
    return " ".join(filtered)


def load_and_prepare_data(nrows_val, words, max_length):
    
    df = pd.read_csv('../Data/train.csv', 
                     header=None, 
                     names=['label','title','review'], 
                     nrows=nrows_val)
    
    df['review_clean'] = df['review'].astype(str).apply(remove_stopwords)
    texts = df['review_clean'].values
    labels = df['label'].values
    labels = labels - 1
    num_classes = len(np.unique(labels))
    labels_one_hot = to_categorical(labels, num_classes=num_classes)

    tokenizer = Tokenizer(num_words=words, oov_token="<DOV>")
    tokenizer.fit_on_texts(texts)

    with open('../Trained_models/Dictionaries/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_length)

    X_train = padded
    Y_train = labels_one_hot

    test_df = pd.read_csv("../Data/test.csv", 
                          header=None, 
                          names=["label", "title", "review"], 
                          nrows=nrows_val)
    
    test_texts = test_df["review"].values
    test_labels = test_df["label"].values
    test_labels = test_labels - 1
    test_labels_one_hot = to_categorical(test_labels, num_classes=num_classes)
    test_sequences = tokenizer.texts_to_sequences(test_texts)
    
    X_val = pad_sequences(test_sequences, maxlen=max_length)
    Y_val = test_labels_one_hot
    
    return X_val, X_train, Y_val, Y_train
