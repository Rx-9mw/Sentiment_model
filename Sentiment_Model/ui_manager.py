import tkinter as tk

class DashboardUI:
    def __init__(self, root, cmd_start_training, cmd_run_manual_test, cmd_run_batch_test, cmd_show_graphs):
        self.root = root
        
        bg_color = "#f8f9fa"
        panel_bg = "#ffffff"
        text_color = "#212529"
        
        btn_primary = "#0d6efd"      
        btn_success = "#198754"      
        btn_warning = "#ffc107"      
        
        btn_primary_hover = "#0b5ed7"
        btn_success_hover = "#157347"
        btn_warning_hover = "#ffca2c"
        
        font_main = ("Segoe UI", 10, "bold")
        font_title = ("Segoe UI", 11, "bold")

        self.root.configure(bg=bg_color)
        
        def bind_hover(widget, default_bg, hover_bg):
            widget.bind("<Enter>", lambda e: widget.config(bg=hover_bg))
            widget.bind("<Leave>", lambda e: widget.config(bg=default_bg))

        top_frame = tk.Frame(root, bg=bg_color)
        top_frame.pack(pady=15, padx=20, fill="x")

        frame_vars = tk.LabelFrame(top_frame, text=" Parameters ", font=font_title, bg=panel_bg, fg=text_color, padx=15, pady=10, relief="solid", borderwidth=1)
        frame_vars.grid(row=0, column=0, padx=10, sticky="nwes")

        tk.Label(frame_vars, text="Number of epochs:", bg=panel_bg, fg=text_color, font=font_main).pack(anchor="w")
        self.input_epoch = tk.Entry(frame_vars, width=15, font=font_main, relief="solid", borderwidth=1)
        self.input_epoch.pack(anchor="w", pady=(2, 12))
        self.input_epoch.insert(0, "6")

        tk.Label(frame_vars, text="Number of neurons (LSTM):", bg=panel_bg, fg=text_color, font=font_main).pack(anchor="w")
        self.input_neurons = tk.Entry(frame_vars, width=15, font=font_main, relief="solid", borderwidth=1)
        self.input_neurons.pack(anchor="w", pady=(2, 12))
        self.input_neurons.insert(0, "16")

        tk.Label(frame_vars, text="Number of neurons (Dense):", bg=panel_bg, fg=text_color, font=font_main).pack(anchor="w")
        self.input_dense = tk.Entry(frame_vars, width=15, font=font_main, relief="solid", borderwidth=1)
        self.input_dense.pack(anchor="w", pady=(2, 12))
        self.input_dense.insert(0, "32")

        tk.Label(frame_vars, text="Embedding:", bg=panel_bg, fg=text_color, font=font_main).pack(anchor="w")
        self.input_embedding = tk.Entry(frame_vars, width=15, font=font_main, relief="solid", borderwidth=1)
        self.input_embedding.pack(anchor="w", pady=(2, 12))
        self.input_embedding.insert(0, "32")

        tk.Label(frame_vars, text="Number of reviews:", bg=panel_bg, fg=text_color, font=font_main).pack(anchor="w")
        self.input_number_of_reviews = tk.Entry(frame_vars, width=15, font=font_main, relief="solid", borderwidth=1)
        self.input_number_of_reviews.pack(anchor="w", pady=(2, 12))
        self.input_number_of_reviews.insert(0, "3200000")

        tk.Label(frame_vars, text="Max length of reviews:", bg=panel_bg, fg=text_color, font=font_main).pack(anchor="w")
        self.input_review_length = tk.Entry(frame_vars, width=15, font=font_main, relief="solid", borderwidth=1)
        self.input_review_length.pack(anchor="w", pady=(2, 12))
        self.input_review_length.insert(0, "150")

        tk.Label(frame_vars, text="Dictionary size:", bg=panel_bg, fg=text_color, font=font_main).pack(anchor="w")
        self.input_dictionary = tk.Entry(frame_vars, width=15, font=font_main, relief="solid", borderwidth=1)
        self.input_dictionary.pack(anchor="w", pady=(2, 12))
        self.input_dictionary.insert(0, "20000")

        tk.Label(frame_vars, text="Dropout rate:", bg=panel_bg, fg=text_color, font=font_main).pack(anchor="w")
        self.input_dropout = tk.Entry(frame_vars, width=15, font=font_main, relief="solid", borderwidth=1)
        self.input_dropout.pack(anchor="w", pady=(2, 12))
        self.input_dropout.insert(0, "0.5")

        self.start_btn = tk.Button(frame_vars, text="Start Training", bg=btn_primary, fg="white", font=("Segoe UI", 10, "bold"), 
                                   relief="flat", cursor="hand2", padx=10, pady=2, command=cmd_start_training)
        self.start_btn.pack(anchor="w", pady=5)
        bind_hover(self.start_btn, btn_primary, btn_primary_hover)

        frame_train = tk.LabelFrame(top_frame, text=" Metrics ", font=font_title, bg=panel_bg, fg=text_color, padx=15, pady=10, relief="solid", borderwidth=1)
        frame_train.grid(row=0, column=1, padx=10, sticky="nwes")

        self.label_epoch = tk.Label(frame_train, text="Epoch: -", bg=panel_bg, fg=text_color, font=font_main)
        self.label_epoch.pack(anchor="w", pady=(0, 10))

        self.label_loss = tk.Label(frame_train, text="Loss Before: - vs After: -", bg=panel_bg, fg=text_color, font=font_main)
        self.label_loss.pack(anchor="w", pady=(0, 10))

        self.label_loss_difference = tk.Label(frame_train, text="Difference: -", bg=panel_bg, fg=text_color, font=("Segoe UI", 10, "bold"))
        self.label_loss_difference.pack(anchor="w", pady=(0, 10))

        frame_val = tk.LabelFrame(top_frame, text=" Raport ", font=font_title, bg=panel_bg, fg=text_color, padx=15, pady=10, relief="solid", borderwidth=1)
        frame_val.grid(row=0, column=2, padx=10, sticky="nwes")

        self.label_val_loss = tk.Label(frame_val, text="Current Validation Loss: -", bg=panel_bg, fg=text_color, font=font_main)
        self.label_val_loss.pack(anchor="w", pady=(0, 5))

        self.label_val_acc = tk.Label(frame_val, text="Current Validation Accuracy: -", bg=panel_bg, fg=text_color, font=font_main)
        self.label_val_acc.pack(anchor="w", pady=(0, 10))

        self.graph_btn = tk.Button(frame_val, text="Show graphs", bg=btn_primary, fg="white", font=("Segoe UI", 9, "bold"), 
                                   relief="flat", cursor="hand2", padx=10, command=cmd_show_graphs)
        self.graph_btn.pack(anchor="w", pady=(0, 10))
        self.graph_btn.config(state="disabled")
        bind_hover(self.graph_btn, btn_primary, btn_primary_hover)

        self.text_report = tk.Text(frame_val, height=12, width=45, font=("Consolas", 9), bg="#f8f9fa", fg=text_color, relief="solid", borderwidth=1)
        self.text_report.pack(anchor="w")
        self.text_report.insert(tk.END, "Start the training to see the results.")
        self.text_report.config(state="disabled")

        middle_frame = tk.LabelFrame(root, text=" Singular text analysis ", font=font_title, bg=panel_bg, fg=text_color, padx=15, pady=15, relief="solid", borderwidth=1)
        middle_frame.pack(pady=10, padx=30, fill="x")

        tk.Label(middle_frame, text="Enter text of the review in english:", bg=panel_bg, fg=text_color, font=font_main).pack(anchor="w")
        
        self.input_test_text = tk.Text(middle_frame, height=3, width=80, font=font_main, borderwidth=1, relief="solid", highlightthickness=2, highlightcolor=btn_primary)
        self.input_test_text.pack(pady=8)

        self.btn_test = tk.Button(middle_frame, text="Check sentiment", bg=btn_success, fg="white", font=("Segoe UI", 10, "bold"), 
                                  relief="flat", cursor="hand2", padx=15, pady=3, command=cmd_run_manual_test)
        self.btn_test.pack(pady=5)
        bind_hover(self.btn_test, btn_success, btn_success_hover)

        self.label_test_result = tk.Label(middle_frame, text="Sentiment: -", bg=panel_bg, fg=text_color, font=("Segoe UI", 12, "bold"))
        self.label_test_result.pack(pady=5)

        bottom_frame = tk.LabelFrame(root, text=" Analyse multiple texts ", font=font_title, bg=panel_bg, fg=text_color, padx=15, pady=15, relief="solid", borderwidth=1)
        bottom_frame.pack(pady=10, padx=30, fill="x")

        tk.Label(bottom_frame, text="Upload a .csv file with text in separate lines. Keep this format for each line: [], \"YOUR TEXT\"", bg=panel_bg, fg=text_color, font=font_main).pack(anchor="w")

        self.btn_batch = tk.Button(bottom_frame, text="Choose a file", bg=btn_warning, fg="#212529", font=("Segoe UI", 10, "bold"), 
                                   relief="flat", cursor="hand2", padx=15, pady=3, command=cmd_run_batch_test)
        self.btn_batch.pack(pady=10)
        bind_hover(self.btn_batch, btn_warning, btn_warning_hover)

        self.label_batch_status = tk.Label(bottom_frame, text="Waiting for a file...", bg=panel_bg, fg=text_color, font=font_main)
        self.label_batch_status.pack(pady=5)