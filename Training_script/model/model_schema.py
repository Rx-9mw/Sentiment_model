
from keras.layers import Bidirectional
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from data.data_processing import load_and_prepare_data
from keras.layers import BatchNormalization

def get_ready(words, max_length):
    nrows_val = 1000
    X_val, X_train, Y_val, Y_train = load_and_prepare_data(nrows_val, words, max_length)

    #hiper-zmienne
    model = Sequential([
        Embedding(input_dim=words, output_dim=32),
        Dropout(0.5),
        Bidirectional(LSTM(32, return_sequences=False)),
        Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        BatchNormalization(),
        Dense(2, activation='softmax')
    ])

    model.build(input_shape=(None, X_train.shape[1]))

    model.summary()

    #opt adam adadelta adagrad rmsprop
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', 
                  metrics=['accuracy'])
    
    return X_val, X_train, Y_val, Y_train, model
