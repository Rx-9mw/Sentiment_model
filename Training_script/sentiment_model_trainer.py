import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, BatchNormalization


df = pd.read_csv('./Data/train.csv', header=None, names=['label','title','review'])

texts = df['review'].values
labels = df['label'].values

labels = labels - 1

num_classes = len(np.unique(labels))

labels_one_hot = to_categorical(labels, num_classes=num_classes)
 
tokenizer = Tokenizer(num_words=1000, oov_token="<DOV>")

tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

padded = pad_sequences(sequences, maxlen=100)

X_train, X_val, Y_train, Y_val = train_test_split(padded, labels_one_hot, test_size=0.2, random_state=42)

model = Sequential([
    Embedding(input_dim=1000, output_dim=32),
    LSTM(64, return_sequences=False),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

# "BEST" ACCURACY MODEL ARCHITECTURE FOR LATER, IT USES A LOT OF THINGS THAT I DONT REALLY UNDERSTAND YET SO LETS PUT THAT AWAY FOR NOW
# model = Sequential([
#     Embedding(input_dim=50000, output_dim=300, input_length=max_len),
#     Bidirectional(LSTM(256, return_sequences=True)),
#     Dropout(0.5),
#     Bidirectional(LSTM(128)),
#     Dropout(0.5),
#     Dense(128, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.5),
#     Dense(64, activation='relu'),
#     Dense(2, activation='softmax')
# ])

model.build(input_shape=(None, padded.shape[1]))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=5, validation_data=(X_val, Y_val))

model.save("./Trained_models/sentiment_please_dont_brick_my_pc_model.keras")
