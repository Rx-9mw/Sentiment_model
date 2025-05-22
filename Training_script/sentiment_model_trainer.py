import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, BatchNormalization


df = pd.read_csv('../Data/converted_reviews.csv', header=None, names=['label','title','review'], nrows=50000)

texts = df['review'].values
labels = df['label'].values

labels = labels - 1

num_classes = len(np.unique(labels))

labels_one_hot = to_categorical(labels, num_classes=num_classes)

tokenizer = Tokenizer(num_words=50000, oov_token="<DOV>")

tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

padded = pad_sequences(sequences, maxlen=150)

X_train, X_val, Y_train, Y_val = train_test_split(padded, labels_one_hot, test_size=0.2, random_state=42)

model = Sequential([
    Embedding(input_dim=50000, output_dim=32),
    LSTM(64, return_sequences=False),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

model.build(input_shape=(None, padded.shape[1]))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=1, validation_data=(X_val, Y_val))

model.save("../Trained_models/sentiment_model.keras")

def predict_review(text):
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(seq, maxlen=150)
    pred = model.predict(padded_seq)
    print("Tokenized sequence:", seq)
    print("Padded input:", padded_seq)
    print("Prediction (probabilities):", pred)
    class_idx = np.argmax(pred)
    return class_idx + 1


print(predict_review("ive got a lamp in the corner of my room behind my desk thats a complete pain in the arse to turn on and off. ive been using this with the lamp for a month now and it works perfectly. added a little velcro and now i have a light switch where ever i want. under my desk, shelf, etc."))
print(predict_review("Very disappointed in this product. It worked perfectly for exactly three days and could not be resuscitated. It was very inexpensive so I did not want to pay half again the price to ship it back for an exchange, so the company would do nothing when they sent me an inquiry as to product satisfaction."))
print(predict_review("This is the all time best book. She mentoins in the book how anyone can be a vampire, who knows. Well anyway I like the idea of having one soulmate and one crazed werewolf crush. P.S I think that it is a good twist on the story about how anone can be part of the night world."))
print(predict_review("I suppose if you were going to sit in the same room and have line-of-sight with the device, lamp, etc. plugged into this control, it might work. Maybe. But I had it completely within the stated range and one basic house wall separating me and the unit, and it was completely unreliable.I'd recommend spending a touch more money and getting a higher quality product."))

print("Correct review : 2 1 2 1")