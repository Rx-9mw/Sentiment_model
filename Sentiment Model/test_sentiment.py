import tensorflow as tf
import numpy as np
import re
import contractions
from tensorflow.keras.layers import Layer

def predict_sentiment(text):
    clean_text = clean_review(text)
    
    vectorized = vectorizer(tf.constant([clean_text]))
    
    pred = model.predict(vectorized)
    class_idx = np.argmax(pred, axis=1)[0]
    
    return "positive" if class_idx == 1 else "negative"

if __name__ == "__main__":
    while True:
        text = input("Wpisz tekst do analizy (lub 'exit' aby wyjść): ")
        if text.lower() == "exit":
            break
        sentiment = predict_sentiment(text)
        print(f"Sentyment: {sentiment}\n")


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


@tf.keras.utils.register_keras_serializable()
class AttentionLayer(Layer):
    def __init__(self, trainable=True, dtype=tf.float32, **kwargs):
        super().__init__(trainable=trainable, dtype=dtype, **kwargs)
        self.supports_masking = True


    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], input_shape[-1]),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[-1],),
            initializer="zeros",
            trainable=True
        )
        self.u = self.add_weight(
            name="att_u",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )
        super().build(input_shape)


    def call(self, inputs):
        score = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)

        return context_vector


model_path = "../Trained_models/Models/sentiment_model_dropout_twice.keras"
vectorizer_path = "../Trained_models/Dictionaries/text_vectorizer.keras"

model = tf.keras.models.load_model(model_path, custom_objects={"AttentionLayer": AttentionLayer})
vectorizer = tf.keras.models.load_model(vectorizer_path)

