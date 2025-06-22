import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

canvas1 = None
canvas2 = None

def create_plot_window(history, root):
    global canvas1, canvas2
    EPOCHS = list(range(1, (len(history.history['loss'])) + 1))
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    # win = tk.Toplevel(root)
    # win.title("CNN Training Metrics")
    root.geometry("900x700")

    # Wykres funkcji Loss czyli błędu uczenia
    if canvas1 is not None:
        canvas1.get_tk_widget().destroy()
        canvas1 = None
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
    