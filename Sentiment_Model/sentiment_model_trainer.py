import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import logging
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
from sklearn.metrics import classification_report, confusion_matrix
import threading
import tkinter as tk
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from ui_manager import DashboardUI
from inference_manager import InferenceManager
from Logger_live import EpochLogger
from plots.plots import create_plot_window
from model.model_schema import get_ready

tf.random.set_seed(42)
np.random.seed(42)

model = None
X_train = None
Y_train = None
X_val = None
Y_val = None
words = None
max_length = None
epochs = None
vectorizer = None

def start_training():
    global training_thread, X_val, X_train, Y_val, Y_train, model, vectorizer

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ui.start_btn.config(state="disabled")
    ui.graph_btn.config(state="disabled")
    ui.text_report.config(state="normal")
    ui.text_report.delete("1.0", tk.END)
    ui.text_report.insert(tk.END, "Training in progress...\n")
    ui.text_report.config(state="disabled")

    assign_variables()
    X_train, Y_train, X_val, Y_val, model, vectorizer = get_ready(
        neurons, 
        dense, 
        embedding, 
        review_num, 
        max_length, 
        words, 
        dropout)

    reset_labels()

    y_int = np.argmax(Y_train, axis=1)
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_int), y=y_int)
    class_weights = dict(enumerate(class_weights))

    lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=1, min_lr=0.00001, verbose=1)

    def train():
        global history, model
        history = model.fit(
            X_train, Y_train, 
            epochs=epochs, 
            batch_size=256, 
            validation_data=(X_val, Y_val),
            class_weight=class_weights, 
            callbacks=[
                epoch_callback, 
                early_stopping, 
                lr_scheduler], 
            verbose=1
        )

    def monitor_completion():
        def is_completed():
            if training_thread.is_alive():
                root.after(500, is_completed)
            else:
                y_pred = model.predict(X_val).argmax(axis=1)
                y_true = Y_val.argmax(axis=1)
                
                report_str = classification_report(y_true, y_pred, zero_division=0)
                cm_str = confusion_matrix(y_true, y_pred)
                full_report = f"RAPORT:\n{report_str}\ERROR MATRIX:\n{cm_str}"
                
                def update_report_ui():
                    ui.text_report.config(state="normal")
                    ui.text_report.delete("1.0", tk.END)
                    ui.text_report.insert(tk.END, full_report)
                    ui.text_report.config(state="disabled")
                    
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
    vectorizer_model.save("../Trained_models/Dictionaries/text_vectorizer.keras")
    
    inference_manager.model = model
    inference_manager.vectorizer = vectorizer
    
    ui.graph_btn.config(state="active")
    ui.start_btn.config(state="active")

def assign_variables():
    global epochs, neurons, dense, embedding, review_num, max_length, words, dropout

    epochs = int(ui.input_epoch.get())
    neurons = int(ui.input_neurons.get())
    dense = int(ui.input_dense.get())
    embedding = int(ui.input_embedding.get())
    review_num = int(ui.input_number_of_reviews.get())
    max_length = int(ui.input_review_length.get())
    words = int(ui.input_dictionary.get())
    dropout = float(ui.input_dropout.get())

def reset_labels():
    ui.label_epoch.config(text="Epoch: -", fg="black")
    ui.label_loss.config(text="Loss Before: - vs After: -", fg="black")
    ui.label_loss_difference.config(text="Difference: -", fg="black")
    ui.label_val_loss.config(text="Current Validation Loss: -", fg="black")
    ui.label_val_acc.config(text="Current Validation Accuracy: -", fg="black")

def show_graphs_wrapper():
    create_plot_window(history, root)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Sentiment Analysis Trainer and Tester.")
    root.geometry("900x1050") 

    inference_manager = InferenceManager(root)

    ui = DashboardUI(root, 
                     cmd_start_training=start_training, 
                     cmd_run_manual_test=inference_manager.run_manual_test, 
                     cmd_run_batch_test=inference_manager.run_batch_test, 
                     cmd_show_graphs=show_graphs_wrapper)

    inference_manager.ui = ui

    epoch_callback = EpochLogger(root, 
                                 ui.label_epoch, 
                                 ui.label_loss, 
                                 ui.label_val_loss, 
                                 ui.label_loss_difference, 
                                 ui.label_val_acc)

    root.mainloop()