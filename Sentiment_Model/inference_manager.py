import tensorflow as tf
import numpy as np
import pandas as pd
import csv
import re
import contractions
import threading
from tkinter import filedialog, messagebox
from tensorflow.keras.layers import Layer

@tf.keras.utils.register_keras_serializable()
class AttentionLayer(Layer):
    def __init__(self, trainable=True, dtype=tf.float32, **kwargs):
        super().__init__(trainable=trainable, dtype=dtype, **kwargs)
        self.supports_masking = True


    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight", 
            shape=(input_shape[-1], 
            input_shape[-1]), 
            initializer="glorot_uniform", 
            trainable=True
            )

        self.b = self.add_weight(
            name="att_bias", 
            shape=(input_shape[-1],), 
            initializer="zeros", 
            trainable=True
            )

        self.u = self.add_weight(
            name="att_u", 
            shape=(input_shape[-1], 1), 
            initializer="glorot_uniform", 
            trainable=True
            )

        super().build(input_shape)


    def call(self, inputs, mask=None):
        score = tf.tanh(
            tf.tensordot(
                inputs, 
                self.W, 
                axes=1
                ) + self.b)

        attention_weights = tf.nn.softmax(
            tf.tensordot(
                score, 
                self.u, 
                axes=1
                ), axis=1)

        context_vector = tf.reduce_sum(
            inputs * attention_weights, 
            axis=1
            )

        return context_vector


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


class InferenceManager:
    def __init__(self, root):
        self.root = root
        self.ui = None
        self.model = None
        self.vectorizer = None


    def ensure_model_loaded(self):
        if self.model is None or self.vectorizer is None:
            try:
                model_path = "../Trained_models/Models/sentiment_model_dropout_twice.keras"
                vectorizer_path = "../Trained_models/Dictionaries/text_vectorizer.keras"
                self.model = tf.keras.models.load_model(model_path, custom_objects={"AttentionLayer": AttentionLayer})
                self.vectorizer = tf.keras.models.load_model(vectorizer_path)
                return True
            except Exception:
                return False
        return True


    def run_manual_test(self):
        if not self.ensure_model_loaded():
            messagebox.showerror("ERROR", "No usable model found!")
            return

        user_text = self.ui.input_test_text.get("1.0", "end-1c")
        if not user_text.strip():
            messagebox.showerror("ERROR", "Input text for analysis!")
            return
            
        clean_text = clean_review(user_text)
        vectorized = self.vectorizer(tf.constant([clean_text]))
        pred = self.model.predict(vectorized)
        class_idx = np.argmax(pred, axis=1)[0]
        
        if class_idx == 1:
            self.ui.label_test_result.config(text="Sentiment: Positive.", fg="green")
        else:
            self.ui.label_test_result.config(text="Sentiment: Negative.", fg="red")

    def run_batch_test(self):
        if not self.ensure_model_loaded():
            messagebox.showerror("ERROR", "No usable model found!")
            return

        input_file = filedialog.askopenfilename(title="Choose a file...", filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")])
        if not input_file:
            return

        output_file = filedialog.asksaveasfilename(title="Save the file as...", defaultextension=".csv", filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")])
        if not output_file:
            return

        def process_file():
            try:
                self.root.after(0, lambda: self.ui.label_batch_status.config(text="Analysing...", fg="blue"))
                
                if input_file.endswith('.xlsx'):
                    df = pd.read_excel(input_file, header=None)
                else:
                    with open(input_file, 'r', encoding='utf-8') as f:
                        data = list(csv.reader(f))
                    df = pd.DataFrame(data)
                
                if df.shape[1] == 1:
                    text_col_idx = 0
                    df.insert(0, 'Sentiment', '')
                    target_col_idx = 'Sentiment'
                else:
                    text_col_idx = 1
                    target_col_idx = 0

                clean_texts = df[text_col_idx].astype(str).apply(clean_review).tolist()
                
                vectorized = self.vectorizer(tf.constant(clean_texts))
                preds = self.model.predict(vectorized)
                class_idxs = np.argmax(preds, axis=1)
                
                labels = ["positive" if idx == 1 else "negative" for idx in class_idxs]
                df[target_col_idx] = "[" + pd.Series(labels) + "]"
                
                if output_file.endswith('.xlsx'):
                    df.to_excel(output_file, index=False, header=False)
                else:
                    df.to_csv(output_file, index=False, header=False)
                    
                self.root.after(0, lambda: self.ui.label_batch_status.config(text=f"Finished! File saved to: {output_file.split('/')[-1]}", fg="green"))
                self.root.after(0, lambda: messagebox.showinfo("Success", "Sentiment analysis has succeeded"))
                
            except Exception as e:
                self.root.after(0, lambda: self.ui.label_batch_status.config(text="ERROR", fg="red"))
                self.root.after(0, lambda err=e: messagebox.showerror("Analysis ERROR", f"Something went wrong:\n{str(err)}"))

        threading.Thread(target=process_file, daemon=True).start()