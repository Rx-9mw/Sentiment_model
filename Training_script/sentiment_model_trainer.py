import pandas as pd
import numpy as np
import pickle
from keras.layers import Bidirectional
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

words = 150000
max_length = 150
epochs = 5

df = pd.read_csv('../Data/train.csv', header=None, names=['label','title','review'])

texts = df['review'].values
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

# X_train, X_val, Y_train, Y_val = train_test_split(padded, labels_one_hot, test_size=0.2, random_state=42)

X_train = padded
Y_train = labels_one_hot

test_df = pd.read_csv("../Data/test.csv", header=None, names=["label", "title", "review"])
test_texts = test_df["review"].values
test_labels = test_df["label"].values
test_labels = test_labels - 1
test_labels_one_hot = to_categorical(test_labels, num_classes=num_classes)

test_sequences = tokenizer.texts_to_sequences(test_texts)
X_val = pad_sequences(test_sequences, maxlen=max_length)
Y_val = test_labels_one_hot

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

model = Sequential([
    Embedding(input_dim=50000, output_dim=16),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.3),
    Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.3),
    Dense(2, activation='softmax')
])

model.build(input_shape=(None, padded.shape[1]))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_val, Y_val), batch_size=64, callbacks=[early_stopping])

model.save("../Trained_models/Models/sentiment_model_dropout_twice.keras")
