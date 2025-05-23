from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
import pickle

max_length = 150
num_of_classes = 2

model = load_model('../Trained_models/Models/sentiment_model_dropout_twice.keras')

with open('../Trained_models/Dictionaries/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

test_df = pd.read_csv("../Data/converted_reviews.csv", header=None, names=["label", "title", "review"])

texts = test_df["review"].values
labels = test_df["label"].values
labels = labels - 1

sequences = tokenizer.texts_to_sequences(texts)

padded = pad_sequences(sequences, maxlen=max_length)

labels_one_hot = to_categorical(labels, num_classes=num_of_classes)

loss, accuracy = model.evaluate(padded, labels_one_hot)

print("\n-----------------------------------------")
print(loss)
print(accuracy)
print("-----------------------------------------\n")

preds = model.predict(padded)
pred_classes = np.argmax(preds, axis=1)
pred_classes += 1

print("\n-----------------------------------------")
print("Sample predictions with reviews:")
for i in range(10):
    print(f"Review {i+1}:")
    print(f"Text      : {texts[i]}")
    print(f"Predicted : {pred_classes[i]}")
    print(f"Actual    : {labels[i] + 1}")
    print("-----------------------------------------\n")



# def predict_review(text):
#     seq = tokenizer.texts_to_sequences([text])
#     padded_seq = pad_sequences(seq, maxlen=max_length)
#     pred = model.predict(padded_seq)
#     print("Prediction (probabilities):", pred)
#     class_idx = np.argmax(pred)
#     print("Picked prediction:", class_idx)
#     return class_idx + 1



# print("-----------------------------------------")
# print("Correct review : 2 | Predicted review : ", predict_review("ive got a lamp in the corner of my room behind my desk thats a complete pain in the arse to turn on and off. ive been using this with the lamp for a month now and it works perfectly. added a little velcro and now i have a light switch where ever i want. under my desk, shelf, etc."))
# print("-----------------------------------------")
# print("Correct review : 1 | Predicted review : ", predict_review("With all due respect to ambient music enthusiasts, I was really disappointed that there was no guitar work whatsoever on this album. Hillage fans of L and Fish Rising be forewarned.Steve Hillage was a pretty darn good guitarist. Maybe L was his showcase with members of Todd Rundgren's Utopia backing him up.Noting that other reviewers have rated this highly, I will give it another listen. However, I am dissapointed in the direction Steve has taken his music."))
# print("-----------------------------------------")
# print("Correct review : 1 | Predicted review : ", predict_review("It worked about 40% of the time when I was standing less than two feet from the outlet. When I moved eight to ten feet away it worked about 5% of the. I would not recommend it."))
# print("-----------------------------------------")
# print("Correct review : 2 | Predicted review : ", predict_review("I liked the film it has what every horror likes in this type of movie the suspense,mystery,good and bad times,plus a shocking ending that makes you think that there is more to story, and plot was very good Robert."))
# print("-----------------------------------------")



