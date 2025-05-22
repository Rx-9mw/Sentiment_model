from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

model = load_model('../Trained_models/sentiment_please_dont_brick_my_pc_model_the_second_one.keras')

texts = ["The author of this book Michael Lane must have been taught how to spell by his son. this stupid son of a bitch mispelled every word in the book. I can understand why he is so dumb mostly because he is crippled, but the fact the books sounds like it was made by the power rangers makes me think this guy is the next michael jackson. in all if you buy this book you are aiding terrorism, and if you read it you are just making yourself that much dumber. SO PLEASE DON'T BUY AND READ THIS BOOK!!!!!!!!!!"]

tokenizer = Tokenizer(num_words=1000, oov_token="<DOV>")

tokenizer.fit_on_texts(texts)
sample_text = ["This was a great product!", "Horrible and slow service.", "It was okay I guess."]
sample_seq = tokenizer.texts_to_sequences(sample_text)
sample_pad = pad_sequences(sample_seq, maxlen=100, padding='post')
predictions = model.predict(sample_pad)
predicted_labels = np.argmax(predictions, axis=1) - 1
print(predicted_labels)


