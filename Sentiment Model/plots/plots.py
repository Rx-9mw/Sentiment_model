import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

def create_plot_window(history, root):
    # 1. Tworzymy nowe, niezależne okno (Toplevel)
    plot_window = tk.Toplevel(root)
    plot_window.title("Wykresy Treningowe - Loss & Accuracy")
    plot_window.geometry("1050x500")
    plot_window.configure(bg="#f4f6f9") # Jasnoszare tło całego okna

    # Zabezpieczenie, aby okno wykresów pojawiło się na wierzchu
    plot_window.attributes('-topmost', True)
    plot_window.focus_force()

    # Pobieranie danych z historii
    EPOCHS = list(range(1, (len(history.history['loss'])) + 1))
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    # Wspólne kolory tła dla spójności
    fig_bg_color = "#f4f6f9"
    ax_bg_color = "#ffffff"

    # ==========================================
    # WYKRES 1: LOSS (Lewa strona nowego okna)
    # ==========================================
    fig1 = plt.Figure(figsize=(5, 4), dpi=100, facecolor=fig_bg_color)
    ax1 = fig1.add_subplot(111, facecolor=ax_bg_color)
    
    # Rysowanie linii (Czerwień i ciemny pomarańcz)
    ax1.plot(EPOCHS, train_loss, color='#e74c3c', marker='o', linewidth=2.5, markersize=6, label='Train Loss')
    ax1.plot(EPOCHS, val_loss, color='#c0392b', marker='s', linestyle='--', linewidth=2.5, markersize=6, label='Validation Loss')
    
    # Stylizacja tekstów i osi
    ax1.set_title("Funkcja Straty (Loss)", fontsize=13, fontweight='bold', color='#2c3e50', pad=15)
    ax1.set_xlabel("Epoka", fontsize=10, color='#34495e')
    ax1.set_ylabel("Loss", fontsize=10, color='#34495e')
    ax1.legend(frameon=True, shadow=True, fancybox=True, borderpad=1)
    ax1.grid(True, linestyle='--', alpha=0.6, color='#bdc3c7')
    
    # Usuwanie zbędnych ramek (Minimalizm)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('#bdc3c7')
    ax1.spines['bottom'].set_color('#bdc3c7')

    canvas1 = FigureCanvasTkAgg(fig1, master=plot_window)
    canvas1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=15, pady=20)
    canvas1.draw()

    # ==========================================
    # WYKRES 2: ACCURACY (Prawa strona nowego okna)
    # ==========================================
    fig2 = plt.Figure(figsize=(5, 4), dpi=100, facecolor=fig_bg_color)
    ax2 = fig2.add_subplot(111, facecolor=ax_bg_color)
    
    # Rysowanie linii (Granat i zieleń)
    ax2.plot(EPOCHS, train_acc, color='#2980b9', marker='o', linewidth=2.5, markersize=6, label='Train Accuracy')
    ax2.plot(EPOCHS, val_acc, color='#27ae60', marker='s', linestyle='--', linewidth=2.5, markersize=6, label='Validation Accuracy')
    
    # Stylizacja tekstów i osi
    ax2.set_title("Dokładność (Accuracy)", fontsize=13, fontweight='bold', color='#2c3e50', pad=15)
    ax2.set_xlabel("Epoka", fontsize=10, color='#34495e')
    ax2.set_ylabel("Accuracy", fontsize=10, color='#34495e')
    ax2.legend(frameon=True, shadow=True, fancybox=True, borderpad=1)
    ax2.grid(True, linestyle='--', alpha=0.6, color='#bdc3c7')
    
    # Usuwanie zbędnych ramek (Minimalizm)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#bdc3c7')
    ax2.spines['bottom'].set_color('#bdc3c7')

    canvas2 = FigureCanvasTkAgg(fig2, master=plot_window)
    canvas2.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=15, pady=20)
    canvas2.draw()