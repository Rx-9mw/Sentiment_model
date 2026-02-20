import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk


def create_plot_window(history, root):
    # 1. Tworzymy nowe, niezależne okno (Toplevel) zamiast rysować po głównym (root)
    plot_window = tk.Toplevel(root)
    plot_window.title("Wykresy Treningowe - Loss & Accuracy")
    plot_window.geometry("1000x500")

    # Zabezpieczenie, aby okno wykresów pojawiło się na wierzchu
    plot_window.attributes('-topmost', True)
    plot_window.focus_force()

    # Pobieranie danych z historii
    EPOCHS = list(range(1, (len(history.history['loss'])) + 1))
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    # ==========================================
    # WYKRES 1: LOSS (Lewa strona nowego okna)
    # ==========================================
    fig1 = plt.Figure(figsize=(5, 4), dpi=100)
    ax1 = fig1.add_subplot(111)
    ax1.plot(EPOCHS, train_loss, marker='o', label='Train Loss')
    ax1.plot(EPOCHS, val_loss, marker='s', label='Validation Loss')
    ax1.set_title("Funkcja Straty (Loss)")
    ax1.set_xlabel("Epoka")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    canvas1 = FigureCanvasTkAgg(fig1, master=plot_window)
    canvas1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH,
                                 expand=True, padx=10, pady=10)
    canvas1.draw()

    # ==========================================
    # WYKRES 2: ACCURACY (Prawa strona nowego okna)
    # ==========================================
    fig2 = plt.Figure(figsize=(5, 4), dpi=100)
    ax2 = fig2.add_subplot(111)
    ax2.plot(EPOCHS, train_acc, marker='o', label='Train Accuracy')
    ax2.plot(EPOCHS, val_acc, marker='s', label='Validation Accuracy')
    ax2.set_title("Dokładność (Accuracy)")
    ax2.set_xlabel("Epoka")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    canvas2 = FigureCanvasTkAgg(fig2, master=plot_window)
    canvas2.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH,
                                 expand=True, padx=10, pady=10)
    canvas2.draw()
