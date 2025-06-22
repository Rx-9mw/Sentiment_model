import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tensorflow.keras.callbacks import EarlyStopping
import tkinter as tk
import threading
from model.model_schema import get_ready
from plots.plots import create_plot_window
from Logger_live import EpochLogger

model = None
X_train = None
Y_train = None
X_val = None
Y_val = None
words = None
max_length = None
epochs = None

root = tk.Tk()
root.geometry("900x400")


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
    

def training_completed():
    model.save("../Trained_models/Models/sentiment_model_dropout_twice.keras")
    graph_btn.config(state="active")
    start_btn.config(state="active")
    
    
def start_training():
    global training_thread, X_val, X_train, Y_val, Y_train, model
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    start_btn.config(state="disabled")
    
    assign_variables()
    X_val, X_train, Y_val, Y_train, model = get_ready(words, max_length)
    reset_labels()
    
    def train():
        global history, model
        history = model.fit(
            X_train, Y_train,
            epochs=epochs,
            validation_data=(X_val, Y_val),
            batch_size=32,
            callbacks=[epoch_callback, early_stopping]
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

metrics = tk.Frame(root)
metrics.pack(pady=20, padx=40)

title_font = ("Helvetica", 12, "bold")

# Header titles for each column (think big vertical HTML divs)
tk.Label(metrics, text="Network Variables", font=title_font).grid(row=0, column=0, padx=30, sticky="w")
tk.Label(metrics, text="Training Metrics", font=title_font).grid(row=0, column=1, padx=30, sticky="w")
tk.Label(metrics, text="Validation Metrics", font=title_font).grid(row=0, column=2, padx=30, sticky="w")

# variables column

label_epoch = tk.Label(metrics, text="Number of epochs", fg="black")
label_epoch.grid(row=1, column=0, sticky="w", padx=30, pady=5)

input_epoch = tk.Entry(metrics, fg="black")
input_epoch.grid(row=2, column=0, sticky="w", padx=30, pady=5)
input_epoch.insert(0, "20")

label_review_length = tk.Label(metrics, text="Max length of reviews", fg="black")
label_review_length.grid(row=3, column=0, sticky="w", padx=30, pady=5)

input_review_length = tk.Entry(metrics, fg="black")
input_review_length.grid(row=4, column=0, sticky="w", padx=30, pady=5)
input_review_length.insert(0, "150")

label_dictionary = tk.Label(metrics, text="Dictionary size", fg="black")
label_dictionary.grid(row=5, column=0, sticky="w", padx=30, pady=5)

input_dictionary = tk.Entry(metrics, fg="black")
input_dictionary.grid(row=6, column=0, sticky="w", padx=30, pady=5)
input_dictionary.insert(0, "100000")

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

graph_btn = tk.Button(metrics, text="Show graphs", bg="lightblue", command=lambda: create_plot_window(history, root))
graph_btn.grid(row=7, column=2, sticky="w", padx=30, pady=5)
graph_btn.config(state="disabled")


epoch_callback = EpochLogger( root, label_epoch, label_loss, label_val_loss, label_loss_difference, label_val_acc)


root.mainloop()