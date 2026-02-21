import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import re
import contractions
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from Logger_live import EpochLogger
from plots.plots import create_plot_window
from model.model_schema import get_ready
import threading
import csv
import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Layer
import pandas as pd


tf.random.set_seed(42)
np.random.seed(42)

# --- ZMIENNE GLOBALNE ---
model = None
X_train = None
Y_train = None
X_val = None
Y_val = None
words = None
max_length = None
epochs = None
vectorizer = None

# --- DEFINICJA WARSTWY ATTENTION ---


@tf.keras.utils.register_keras_serializable()
class AttentionLayer(Layer):
    def __init__(self, trainable=True, dtype=tf.float32, **kwargs):
        super().__init__(trainable=trainable, dtype=dtype, **kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1]),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[-1],),
                                 initializer="zeros", trainable=True)
        self.u = self.add_weight(name="att_u", shape=(input_shape[-1], 1),
                                 initializer="glorot_uniform", trainable=True)
        super().build(input_shape)

    def call(self, inputs, mask=None):
        score = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(
            tf.tensordot(score, self.u, axes=1), axis=1)
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)
        return context_vector

# --- FUNKCJE POMOCNICZE I TESTUJĄCE ---


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


def ensure_model_loaded():
    global model, vectorizer
    if model is None or vectorizer is None:
        try:
            model_path = "../Trained_models/Models/sentiment_model_dropout_twice.keras"
            vectorizer_path = "../Trained_models/Dictionaries/text_vectorizer.keras"
            model = tf.keras.models.load_model(model_path, custom_objects={
                                               "AttentionLayer": AttentionLayer})
            vectorizer = tf.keras.models.load_model(vectorizer_path)
            return True
        except Exception as e:
            return False
    return True


def run_manual_test():
    if not ensure_model_loaded():
        label_test_result.config(
            text="Błąd: Najpierw wytrenuj model!", fg="red")
        return

    user_text = input_test_text.get("1.0", "end-1c")
    if not user_text.strip():
        label_test_result.config(
            text="Wynik: Wpisz tekst do analizy!", fg="orange")
        return

    clean_text = clean_review(user_text)
    vectorized = vectorizer(tf.constant([clean_text]))
    pred = model.predict(vectorized)
    class_idx = np.argmax(pred, axis=1)[0]

    if class_idx == 1:
        label_test_result.config(
            text="Wynik: Sentyment POZYTYWNY 😊", fg="green")
    else:
        label_test_result.config(text="Wynik: Sentyment NEGATYWNY 😞", fg="red")


def run_batch_test():
    if not ensure_model_loaded():
        messagebox.showerror("Błąd", "Nie znaleziono modelu! Najpierw wytrenuj model.")
        return

    input_file = filedialog.askopenfilename(title="Wybierz plik z recenzjami", 
                                            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")])
    if not input_file:
        return

    output_file = filedialog.asksaveasfilename(title="Zapisz wynik jako", 
                                               defaultextension=".csv", 
                                               filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")])
    if not output_file:
        return

    def process_file():
        try:
            root.after(0, lambda: label_batch_status.config(text="Przetwarzanie pliku...", fg="blue"))
            
            # 1. Inteligentne wczytywanie (obsługuje przecinki i średniki)
            # 1. Kuloodporne wczytywanie (omija błędy cudzysłowów Pandasa)
            if input_file.endswith('.xlsx'):
                df = pd.read_excel(input_file, header=None)
            else:
                with open(input_file, 'r', encoding='utf-8') as f:
                    data = list(csv.reader(f))
                df = pd.DataFrame(data)
            
            # 2. Elastyczne dopasowanie kolumn (nie potrzebujesz już pustych nawiasów [])
            if df.shape[1] == 1:
                # Jeśli jest tylko jedna kolumna z tekstem, to ją analizujemy
                text_col_idx = 0
                df.insert(0, 'Sentyment', '') # Program sam dorobi pierwszą kolumnę na wyniki
                target_col_idx = 'Sentyment'
            else:
                # Jeśli są dwie kolumny, tekst jest w drugiej
                text_col_idx = 1
                target_col_idx = 0

            clean_texts = df[text_col_idx].astype(str).apply(clean_review).tolist()
            
            # Podgląd w terminalu, żebyś miał dowód, że analizuje NOWE dane
            print(f"\n--- BATCH TEST ---")
            print(f"Analizuję {len(clean_texts)} recenzji z pliku: {input_file.split('/')[-1]}")
            
            # 3. Predykcja
            vectorized = vectorizer(tf.constant(clean_texts))
            preds = model.predict(vectorized)
            class_idxs = np.argmax(preds, axis=1)
            
            # 4. Zapisywanie
            labels = ["pozytywna" if idx == 1 else "negatywna" for idx in class_idxs]
            df[target_col_idx] = "[" + pd.Series(labels) + "]"
            
            if output_file.endswith('.xlsx'):
                df.to_excel(output_file, index=False, header=False)
            else:
                df.to_csv(output_file, index=False, header=False)
                
            # Sukces (używamy root.after, aby okienko pokazało się poprawnie z wątku)
            root.after(0, lambda: label_batch_status.config(text=f"Gotowe! Zapisano do: {output_file.split('/')[-1]}", fg="green"))
            root.after(0, lambda: messagebox.showinfo("Sukces", "Przetwarzanie zakończone. Otwórz nowy plik!"))
            
        except Exception as e:
            # Pokaże dokładną przyczynę błędu na ekranie
            root.after(0, lambda: label_batch_status.config(text="Wystąpił błąd!", fg="red"))
            root.after(0, lambda err=e: messagebox.showerror("Błąd przetwarzania", f"Coś poszło nie tak:\n{str(err)}"))

    threading.Thread(target=process_file, daemon=True).start()

# --- FUNKCJE TRENINGOWE ---


def start_training():
    global training_thread, X_val, X_train, Y_val, Y_train, model, vectorizer

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True)
    start_btn.config(state="disabled")
    graph_btn.config(state="disabled")
    text_report.config(state="normal")
    text_report.delete("1.0", tk.END)
    text_report.insert(tk.END, "Trening w toku...\nCzekaj na wyniki.")
    text_report.config(state="disabled")

    assign_variables()
    X_train, Y_train, X_val, Y_val, model, vectorizer = get_ready(
        words, max_length)

    reset_labels()

    y_int = np.argmax(Y_train, axis=1)
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_int), y=y_int)
    class_weights = dict(enumerate(class_weights))

    lr_scheduler = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=1, min_lr=0.00001, verbose=1)

    def train():
        global history, model
        history = model.fit(
            X_train, Y_train, epochs=epochs, batch_size=32, validation_data=(X_val, Y_val),
            class_weight=class_weights, callbacks=[
                epoch_callback, early_stopping, lr_scheduler], verbose=1
        )

    def monitor_completion():
        def is_completed():
            if training_thread.is_alive():
                root.after(500, is_completed)
            else:
                # Obliczanie metryk po zakończeniu treningu
                y_pred = model.predict(X_val).argmax(axis=1)
                y_true = Y_val.argmax(axis=1)

                report_str = classification_report(y_true, y_pred)
                cm_str = confusion_matrix(y_true, y_pred)

                # Formatowanie tekstu do wyświetlenia
                full_report = f"RAPORT KLASYFIKACJI:\n{report_str}\nMACIERZ POMYŁEK:\n{cm_str}"

                # Aktualizacja pola tekstowego w GUI
                def update_report_ui():
                    text_report.config(state="normal")
                    text_report.delete("1.0", tk.END)
                    text_report.insert(tk.END, full_report)
                    text_report.config(state="disabled")

                root.after(0, update_report_ui)
                training_completed()

        is_completed()

    training_thread = threading.Thread(target=train, daemon=True)
    training_thread.start()
    monitor_completion()


def training_completed():
    model.save("../Trained_models/Models/sentiment_model_dropout_twice.keras")
    vectorizer_model = tf.keras.Sequential([vectorizer])
    _ = vectorizer_model(tf.constant(["dummy text"]))
    vectorizer_model.save(
        "../Trained_models/Dictionaries/text_vectorizer.keras")

    graph_btn.config(state="active")
    start_btn.config(state="active")


def assign_variables():
    global words, max_length, epochs, X_val, Y_val
    epochs = int(input_epoch.get())
    max_length = int(input_review_length.get())
    words = int(input_dictionary.get())


def reset_labels():
    label_epoch.config(text="Epoch: -", fg="black")
    label_loss.config(text="Loss Before: - vs After: -", fg="black")
    label_loss_difference.config(text="Difference: -", fg="black")
    label_val_loss.config(text="Current Validation Loss: -", fg="black")
    label_val_acc.config(text="Current Validation Accuracy: -", fg="black")


# ==========================================
# BUDOWA INTERFEJSU
# ==========================================
root = tk.Tk()
root.title("Sentiment Analysis Dashboard")
root.geometry("950x750")

# --- PANEL 1: TRENING (Góra) ---
top_frame = tk.Frame(root)
top_frame.pack(pady=10, padx=20, fill="x")

# Parametry
frame_vars = tk.LabelFrame(top_frame, text="Parametry Sieci", font=(
    "Helvetica", 11, "bold"), padx=15, pady=10)
frame_vars.grid(row=0, column=0, padx=10, sticky="nwes")

tk.Label(frame_vars, text="Number of epochs:").pack(anchor="w")
input_epoch = tk.Entry(frame_vars, width=15)
input_epoch.pack(anchor="w", pady=(0, 10))
input_epoch.insert(0, "6")

tk.Label(frame_vars, text="Max length of reviews:").pack(anchor="w")
input_review_length = tk.Entry(frame_vars, width=15)
input_review_length.pack(anchor="w", pady=(0, 10))
input_review_length.insert(0, "150")

tk.Label(frame_vars, text="Dictionary size:").pack(anchor="w")
input_dictionary = tk.Entry(frame_vars, width=15)
input_dictionary.pack(anchor="w", pady=(0, 10))
input_dictionary.insert(0, "20000")

start_btn = tk.Button(frame_vars, text="Start Training",
                      bg="lightblue", command=start_training)
start_btn.pack(anchor="w", pady=5)

# Metryki na żywo
frame_train = tk.LabelFrame(top_frame, text="Metryki Treningowe", font=(
    "Helvetica", 11, "bold"), padx=15, pady=10)
frame_train.grid(row=0, column=1, padx=10, sticky="nwes")

label_epoch = tk.Label(frame_train, text="Epoch: -",
                       fg="black", font=("Helvetica", 10))
label_epoch.pack(anchor="w", pady=(0, 10))

label_loss = tk.Label(frame_train, text="Loss Before: - vs After: -",
                      fg="black", font=("Helvetica", 10))
label_loss.pack(anchor="w", pady=(0, 10))

label_loss_difference = tk.Label(
    frame_train, text="Difference: -", fg="black", font=("Helvetica", 10, "bold"))
label_loss_difference.pack(anchor="w", pady=(0, 10))

# Walidacja i Raport
frame_val = tk.LabelFrame(top_frame, text="Metryki Walidacyjne i Raport", font=(
    "Helvetica", 11, "bold"), padx=15, pady=10)
frame_val.grid(row=0, column=2, padx=10, sticky="nwes")

label_val_loss = tk.Label(
    frame_val, text="Current Validation Loss: -", fg="black", font=("Helvetica", 10))
label_val_loss.pack(anchor="w", pady=(0, 5))

label_val_acc = tk.Label(
    frame_val, text="Current Validation Accuracy: -", fg="black", font=("Helvetica", 10))
label_val_acc.pack(anchor="w", pady=(0, 10))

graph_btn = tk.Button(frame_val, text="Show graphs", bg="lightblue",
                      command=lambda: create_plot_window(history, root))
graph_btn.pack(anchor="w", pady=(0, 10))
graph_btn.config(state="disabled")

# Nowe pole tekstowe na raport końcowy z terminala
text_report = tk.Text(frame_val, height=13, width=45,
                      font=("Courier", 8), bg="#f4f4f4")
text_report.pack(anchor="w")
text_report.insert(tk.END, "Brak wyników. Rozpocznij trening.")
text_report.config(state="disabled")

epoch_callback = EpochLogger(
    root, label_epoch, label_loss, label_val_loss, label_loss_difference, label_val_acc)

# --- PANEL 2: TESTOWANIE RĘCZNE (Środek) ---
middle_frame = tk.LabelFrame(root, text="Ręczne Testowanie Modelu (Pojedyncza opinia)", font=(
    "Helvetica", 11, "bold"), padx=15, pady=10)
middle_frame.pack(pady=10, padx=30, fill="x")

tk.Label(middle_frame, text="Wpisz recenzję po angielsku:",
         font=("Helvetica", 10)).pack(anchor="w")
input_test_text = tk.Text(middle_frame, height=3,
                          width=80, font=("Helvetica", 10))
input_test_text.pack(pady=5)

btn_test = tk.Button(middle_frame, text="Sprawdź Sentyment", bg="lightgreen", font=(
    "Helvetica", 10, "bold"), command=run_manual_test)
btn_test.pack(pady=5)

label_test_result = tk.Label(
    middle_frame, text="Wynik: -", font=("Helvetica", 12, "bold"))
label_test_result.pack(pady=5)

# --- PANEL 3: BATCH PROCESSING (Dół) ---
bottom_frame = tk.LabelFrame(root, text="Analiza Zbiorcza (Wgraj plik z recenzjami)", font=(
    "Helvetica", 11, "bold"), padx=15, pady=10)
bottom_frame.pack(pady=10, padx=30, fill="x")

tk.Label(bottom_frame, text="Wgraj plik CSV/Excel. Program oczekuje, że w 2. kolumnie (Index 1) znajduje się tekst recenzji.",
         font=("Helvetica", 10)).pack(anchor="w")

btn_batch = tk.Button(bottom_frame, text="Wybierz plik i analizuj", bg="gold", font=(
    "Helvetica", 10, "bold"), command=run_batch_test)
btn_batch.pack(pady=10)

label_batch_status = tk.Label(
    bottom_frame, text="Status: Oczekiwanie na plik...", font=("Helvetica", 10))
label_batch_status.pack(pady=5)

root.mainloop()
