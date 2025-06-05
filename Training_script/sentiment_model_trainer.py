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
import tkinter as tk
# from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from nltk.corpus import stopwords
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

#nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    tokens = text.split()
    filtered = [w for w in tokens if w.lower() not in stop_words]
    return " ".join(filtered)

words = 100000
max_length = 150
epochs = 50


df = pd.read_csv('../Data/train.csv', header=None, names=['label','title','review'], nrows=1000000)

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

X_train, X_val, Y_train, Y_val = train_test_split(padded, labels_one_hot, test_size=0.1, random_state=42)

# X_train = padded
# Y_train = labels_one_hot

# test_df = pd.read_csv("../Data/test.csv", header=None, names=["label", "title", "review"], nrows=100000)
# test_texts = test_df["review"].values
# test_labels = test_df["label"].values
# test_labels = test_labels - 1
# test_labels_one_hot = to_categorical(test_labels, num_classes=num_classes)

# test_sequences = tokenizer.texts_to_sequences(test_texts)
# X_val = pad_sequences(test_sequences, maxlen=max_length)
# Y_val = test_labels_one_hot

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=6,
    restore_best_weights=True
)

#hiper-zmienne
model = Sequential([
    Embedding(input_dim=words, output_dim=32, embeddings_regularizer=regularizers.l2(1e-6)),
    Dropout(0.5),
    Bidirectional(LSTM(16, return_sequences=False)),
    Dropout(0.5),
    Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    BatchNormalization(),
    Dense(2, activation='softmax')
])

model.build(input_shape=(None, padded.shape[1]))

model.summary()

opt = Adam(learning_rate=0.0005)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

#opt adam adadelta adagrad rmsprop
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_val, Y_val), batch_size=32, callbacks=[early_stopping, reduce_lr] ) 


model.save("../Trained_models/Models/sentiment_model_dropout_twice.keras")

'''
Program tworzenia wykresów do kontroli uczenia SSN/DNN. Wersja anglojęzyczna z bardzo uproszczonym interfejsem GUI.
Na użytek Studentów mojej grupy seminaryjnej.
(c) 2025 Krzysztof Michalik, Uniwersytet WSB Merito Chorzów/Katowice
'''

    # Symulowane dane dla 5 epok, można je łatwo zastapić realnymi danymi
EPOCHS = list(range(1, (len(history.history['loss'])) + 1))
train_loss = history.history['loss']
val_loss   = history.history['val_loss']
train_acc  = history.history['accuracy']
val_acc    = history.history['val_accuracy'] # symuluje overfitting

def create_plot_window():
    win = tk.Toplevel(root)
    win.title("CNN Training Metrics")
    win.geometry("900x400")

    # Wykres funkcji Loss czyli błędu uczenia
    fig1 = plt.Figure(figsize=(4.5, 3), dpi=100)
    ax1 = fig1.add_subplot(111)
    ax1.plot(EPOCHS, train_loss, marker='o', label='Train Loss')
    ax1.plot(EPOCHS, val_loss, marker='s', label='Validation Loss')
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)
    canvas1 = FigureCanvasTkAgg(fig1, master=win)
    canvas1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    canvas1.draw()

    # Wykres Accuracy dokładnosci uczenia - w uproszczeniu  procent poprawnie sklasyfikowanych przykładów (zarówno pozytywnych, 
    # jak i negatywnych)
    fig2 = plt.Figure(figsize=(4.5, 3), dpi=100)
    ax2 = fig2.add_subplot(111)
    ax2.plot(EPOCHS, train_acc, marker='o', label='Train Accuracy')
    ax2.plot(EPOCHS, val_acc, marker='s', label='Validation Accuracy')
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)
    canvas2 = FigureCanvasTkAgg(fig2, master=win)
    canvas2.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    canvas2.draw()

    # Okno główne
root = tk.Tk()
root.title("Educational CNN Training Visualizer")
root.geometry("300x150")

btn = tk.Button(root, text="Show Training Charts", command=create_plot_window, bg="lightblue")
btn.pack(pady=30)

root.mainloop()
