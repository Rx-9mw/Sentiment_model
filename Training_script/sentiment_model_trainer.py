import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
import numpy as np
import pickle
from keras.layers import Bidirectional
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from nltk.corpus import stopwords
from keras.layers import BatchNormalization
from tensorflow.keras.callbacks import Callback
import threading


# #nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    tokens = text.split()
    filtered = [w for w in tokens if w.lower() not in stop_words]
    return " ".join(filtered)

words = None
max_length = None
epochs = None
canvas1 = None
canvas2 = None
model = None
X_train = None
Y_train = None
X_val = None
Y_val = None

def get_ready():
    global model, X_train, Y_train, X_val, Y_val
    df = pd.read_csv('../Data/train.csv', header=None, names=['label','title','review'], nrows=1000)

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

    # X_train, X_val, Y_train, Y_val = train_test_split(padded, labels_one_hot, test_size=0.2, random_state=42)

    X_train = padded
    Y_train = labels_one_hot

    test_df = pd.read_csv("../Data/test.csv", header=None, names=["label", "title", "review"], nrows=10000)
    test_texts = test_df["review"].values
    test_labels = test_df["label"].values
    test_labels = test_labels - 1
    test_labels_one_hot = to_categorical(test_labels, num_classes=num_classes)

    test_sequences = tokenizer.texts_to_sequences(test_texts)
    X_val = pad_sequences(test_sequences, maxlen=max_length)
    Y_val = test_labels_one_hot

    # early_stopping = EarlyStopping(
    #     monitor='val_loss',
    #     patience=5,
    #     restore_best_weights=True
    # )

    #hiper-zmienne
    model = Sequential([
        Embedding(input_dim=words, output_dim=32),
        Dropout(0.5),
        Bidirectional(LSTM(32, return_sequences=False)),
        Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        BatchNormalization(),
        Dense(2, activation='softmax')
    ])

    model.build(input_shape=(None, padded.shape[1]))

    model.summary()

    #opt adam adadelta adagrad rmsprop
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


def create_plot_window():
    global canvas1, canvas2
    EPOCHS = list(range(1, (len(history.history['loss'])) + 1))
    train_loss = history.history['loss']
    val_loss   = history.history['val_loss']
    train_acc  = history.history['accuracy']
    val_acc    = history.history['val_accuracy'] # symuluje overfitting
    # win = tk.Toplevel(root)
    # win.title("CNN Training Metrics")
    root.geometry("900x700")

    # Wykres funkcji Loss czyli błędu uczenia
    if canvas1 is not None:
        canvas1.get_tk_widget().destroy()
        canvas1 = None
        print("deleted 1")
    fig1 = plt.Figure(figsize=(4.5, 3), dpi=100)
    ax1 = fig1.add_subplot(111)
    ax1.plot(EPOCHS, train_loss, marker='o', label='Train Loss')
    ax1.plot(EPOCHS, val_loss, marker='s', label='Validation Loss')
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)
    canvas1 = FigureCanvasTkAgg(fig1, master=root)
    canvas1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    canvas1.draw()

    # Wykres Accuracy dokładnosci uczenia - w uproszczeniu  procent poprawnie sklasyfikowanych przykładów (zarówno pozytywnych, 
    # jak i negatywnych)
    if canvas2 is not None:
        canvas2.get_tk_widget().destroy()
        canvas2 = None
        print("deleted 2")
    fig2 = plt.Figure(figsize=(4.5, 3), dpi=100)
    ax2 = fig2.add_subplot(111)
    ax2.plot(EPOCHS, train_acc, marker='o', label='Train Accuracy')
    ax2.plot(EPOCHS, val_acc, marker='s', label='Validation Accuracy')
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)
    canvas2 = FigureCanvasTkAgg(fig2, master=root)
    canvas2.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    canvas2.draw()

def training_completed():
    model.save("../Trained_models/Models/sentiment_model_dropout_twice.keras")
    graph_btn.config(state="active")
    start_btn.config(state="active")
    
def reset_labels():
    label_epoch.config(text="Epoch: -", fg="black")
    label_loss.config(text="Loss Before: - vs After: -", fg="black")
    label_loss_difference.config(text="Difference: -", fg="black")
    label_val_loss.config(text="Current Validation Loss: -", fg="black")
    label_val_acc.config(text="Current Validation Accuracy: -", fg="black")

root = tk.Tk()
root.geometry("900x400")

class EpochLogger(Callback):
    def __init__(self, root, label_epoch, label_loss, label_val_loss, label_loss_difference, label_val_acc):
        super().__init__()
        self.root = root
        self.label_epoch = label_epoch
        self.label_loss = label_loss
        self.label_val_loss = label_val_loss
        self.label_loss_difference = label_loss_difference
        self.label_val_acc = label_val_acc
        self.loss = None
        self.loss_before = None
        self.epoch_count = 0
    
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_count = epoch + 1
        self.root.after(0, lambda: self.label_epoch.config(text=f"Working on Epoch {epoch + 1}..."))
        
    def on_train_end(self, logs=None):
        self.root.after(0, lambda: self.label_epoch.config(text=f"Finished all {self.epoch_count} Epochs."))
        
    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get("val_loss")
        def progress():
            if epoch > 0:
                if self.loss is not None: 
                    if current_loss < self.loss:
                        return "green", str(round(current_loss - self.loss, 4))
                    else:
                        return "red", "+" + str(round(current_loss - self.loss, 4))
                    
            return "blue", "No difference"
        
        color, difference = progress()
        last_loss = self.loss
        self.loss = current_loss
        val_loss = logs.get("val_loss")
        val_accuracy = logs.get("val_accuracy")
        if last_loss is not None:
            self.root.after(0, lambda: self.label_loss.config(text=f"Loss Before: {last_loss:.4f} vs After: {current_loss:.4f}"))
        else:
            self.root.after(0, lambda: self.label_loss.config(text=f"Loss Before: - vs After: {current_loss:.4f}"))
            
        self.root.after(0, lambda: self.label_loss_difference.config(text=f"Difference: {difference}", fg=color))
        self.root.after(0, lambda: self.label_val_loss.config(text=f"Current Validation Loss: {val_loss:.4f}"))
        self.root.after(0, lambda: self.label_val_acc.config(text=f"Current Validation Accuracy: {val_accuracy:.4f}"))
      
def assign_variables():
    global words, max_length, epochs
    epochs = int(input_epoch.get())
    max_length = int(input_review_length.get())
    words = int(input_dictionary.get())
        
def start_training():
    global training_thread
    graph_btn.config(state="disabled")
    start_btn.config(state="disabled")
    
    assign_variables()
    get_ready()
    reset_labels()
    
    def train():
        global history
        history = model.fit(
            X_train, Y_train,
            epochs=epochs,
            validation_data=(X_val, Y_val),
            batch_size=32,
            callbacks=[epoch_callback]
        )
    
    def monitor_completion():
        def is_completed():
            if training_thread.is_alive():
                root.after(500, is_completed)
            else:
                training_completed()
        
        is_completed()
            
    training_thread = threading.Thread(target=train, daemon=True)
    training_thread.start()
    monitor_completion()



# '''
# Program tworzenia wykresów do kontroli uczenia SSN/DNN. Wersja anglojęzyczna z bardzo uproszczonym interfejsem GUI.
# Na użytek Studentów mojej grupy seminaryjnej.
# (c) 2025 Krzysztof Michalik, Uniwersytet WSB Merito Chorzów/Katowice
# '''

#     # Symulowane dane dla 5 epok, można je łatwo zastapić realnymi danymi

metrics = tk.Frame(root)
metrics.pack(pady=20, padx=40)

title_font = ("Helvetica", 12, "bold")

# Header titles for each column
tk.Label(metrics, text="Network Variables", font=title_font).grid(row=0, column=0, padx=30, sticky="w")
tk.Label(metrics, text="Training Metrics", font=title_font).grid(row=0, column=1, padx=30, sticky="w")
tk.Label(metrics, text="Validation Metrics", font=title_font).grid(row=0, column=2, padx=30, sticky="w")

# variables column

label_epoch = tk.Label(metrics, text="Number of epochs", fg="black")
label_epoch.grid(row=1, column=0, sticky="w", padx=30, pady=5)

input_epoch = tk.Entry(metrics, fg="black")
input_epoch.grid(row=2, column=0, sticky="w", padx=30, pady=5)
input_epoch.insert(0, "2")

label_review_length = tk.Label(metrics, text="Max length of reviews", fg="black")
label_review_length.grid(row=3, column=0, sticky="w", padx=30, pady=5)

input_review_length = tk.Entry(metrics, fg="black")
input_review_length.grid(row=4, column=0, sticky="w", padx=30, pady=5)
input_review_length.insert(0, "150")

label_dictionary = tk.Label(metrics, text="Dictionary size", fg="black")
label_dictionary.grid(row=5, column=0, sticky="w", padx=30, pady=5)

input_dictionary = tk.Entry(metrics, fg="black")
input_dictionary.grid(row=6, column=0, sticky="w", padx=30, pady=5)
input_dictionary.insert(0, "1000")

start_btn = tk.Button(metrics, text="Start Training", bg="lightblue", command=start_training)
start_btn.grid(row=7, column=0, sticky="w", padx=30, pady=5)

# differences column
label_epoch = tk.Label(metrics, text="Epoch: -", fg="black")
label_epoch.grid(row=2, column=1, sticky="w", padx=30, pady=5)

label_loss = tk.Label(metrics, text="Loss Before: - vs After: ", fg="black")
label_loss.grid(row=4, column=1, sticky="w", padx=30, pady=5)

label_loss_difference = tk.Label(metrics, text="Difference: -", fg="black")
label_loss_difference.grid(row=6, column=1, sticky="w", padx=30, pady=5)

# validation column
label_val_loss = tk.Label(metrics, text="Current Validation Loss: -", fg="black")
label_val_loss.grid(row=2, column=2, sticky="w", padx=30, pady=5)

label_val_acc = tk.Label(metrics, text="Current Validation Accuracy: -", fg="black")
label_val_acc.grid(row=4, column=2, sticky="w", padx=30, pady=5)

graph_btn = tk.Button(metrics, text="Show graphs", bg="lightblue", command=create_plot_window)
graph_btn.grid(row=7, column=2, sticky="w", padx=30, pady=5)


epoch_callback = EpochLogger( root, label_epoch, label_loss, label_val_loss, label_loss_difference, label_val_acc)


root.mainloop()