from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

model = load_model('../Trained_models/sentiment_model.keras')

texts = ["The author of this book Michael Lane must have been taught how to spell by his son. this stupid son of a bitch mispelled every word in the book. I can understand why he is so dumb mostly because he is crippled, but the fact the books sounds like it was made by the power rangers makes me think this guy is the next michael jackson. in all if you buy this book you are aiding terrorism, and if you read it you are just making yourself that much dumber. SO PLEASE DON'T BUY AND READ THIS BOOK!!!!!!!!!!"]

tokenizer = Tokenizer(num_words=1000, oov_token="<DOV>")

def predict_review(text):
    tokenizer.fit_on_texts(texts)
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(seq, maxlen=150, padding='post')
    pred = model.predict(padded_seq)
    print("Tokenized sequence:", seq)
    print("Padded input:", padded_seq)
    print("Prediction (probabilities):", pred)
    class_idx = np.argmax(pred)
    return class_idx + 1

print(predict_review(texts[0]))


