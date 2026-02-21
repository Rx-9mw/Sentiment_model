from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout,
    Bidirectional,
    GlobalAveragePooling1D, Input
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import Precision, Recall
from data.data_processing import load_and_prepare_data
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable
import tensorflow as tf


def get_ready(neurons, dense, embedding, nrows, max_length, max_tokens, dropout):
    X_train, y_train, X_val, y_val, vectorizer = load_and_prepare_data(
        nrows, max_tokens, max_length
    )

    vocab_size = len(vectorizer.get_vocabulary())

    inputs = Input(shape=(max_length,))

    x = Embedding(
        input_dim=vocab_size,
        output_dim=embedding,
        mask_zero=False
    )(inputs)
    x = Bidirectional(
        LSTM(
            neurons,
            return_sequences=True,
            recurrent_dropout=0.2
        ))(x)
    x = AttentionLayer()(x)
    x = Dense(
        dense,
        activation="relu",
        kernel_regularizer=l2(0.001)
    )(x)
    x = Dropout(dropout)(x)

    outputs = Dense(2, activation="softmax")(x)
    model = Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            Precision(name="precision"),
            Recall(name="recall")
        ]
    )

    model.summary()

    return X_train, y_train, X_val, y_val, model, vectorizer


@register_keras_serializable()
class AttentionLayer(Layer):
    def __init__(self, trainable=True, dtype=tf.float32, **kwargs):
        super(AttentionLayer, self).__init__(
            trainable=trainable, dtype=dtype, **kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return None

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

    def call(self, inputs, mask=None):
        # inputs: (batch, time, features)
        score = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(
            tf.tensordot(score, self.u, axes=1), axis=1)
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)

        return context_vector

    def get_config(self):
        config = super().get_config()
        return config
