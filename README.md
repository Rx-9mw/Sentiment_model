# Sentiment Analysis with BiLSTM and Attention Mechanism

![Python](https://img.shields.io/badge/Python-3.9-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-Educational-green)

## Overview

This project implements a deep learning-based sentiment analysis system for product reviews using a Bidirectional Long Short-Term Memory (BiLSTM) network enhanced with a custom Attention mechanism.

The system was developed as part of an Engineering Thesis and focuses on binary sentiment classification (Positive / Negative) of product reviews from the Amazon Reviews Dataset.

The project includes:

* Data preprocessing pipeline
* Custom Attention Layer implementation
* BiLSTM sentiment classification model
* Training and evaluation tools
* Desktop GUI built with Tkinter
* Model persistence and loading

---

## Key Features

✅ Bidirectional LSTM architecture

✅ Custom Attention Mechanism

✅ Large-scale dataset support (Amazon Reviews)

✅ Automatic dataset balancing

✅ Early Stopping and Learning Rate Scheduling

✅ L2 Regularization and Dropout

✅ Graphical User Interface

✅ Reproducible training process (Seed = 42)

---

## Model Architecture

```text
Input
 │
 ▼
Embedding Layer
 │
 ▼
Bidirectional LSTM
 │
 ▼
Attention Layer
 │
 ▼
Dense (ReLU + L2)
 │
 ▼
Dropout
 │
 ▼
Softmax Output
```

### Architecture Details

| Layer               | Configuration |
| ------------------- | ------------- |
| Input Length        | Configurable  |
| Embedding Dimension | 150           |
| LSTM Units          | 16            |
| Recurrent Dropout   | 0.2           |
| Dense Units         | 32            |
| L2 Regularization   | 0.001         |
| Dropout             | 0.5           |
| Output Classes      | 2             |
| Output Activation   | Softmax       |

---

## Technologies

### Machine Learning

* TensorFlow
* Keras
* NumPy
* Scikit-learn

### Data Processing

* Pandas
* NLTK
* Regex
* Contractions

### Application Layer

* Python 3.9
* Tkinter
* Threading

---

## Dataset

The model was trained using a subset of the Amazon Reviews Dataset.

### Dataset Characteristics

* Product reviews in English
* Over 3.2 million reviews used for training
* Binary sentiment labels
* CSV format

### Preprocessing Pipeline

1. Lowercase conversion
2. HTML tag removal
3. URL removal
4. Contraction expansion
5. Non-ASCII character filtering
6. Punctuation removal
7. Stopword removal
8. Tokenization
9. Vectorization

---

## Training Strategy

### Regularization

```python
Dropout = 0.5
Recurrent Dropout = 0.2
L2 = 0.001
```

### Optimization

* Adam Optimizer
* ReduceLROnPlateau
* EarlyStopping
* Class Weight Balancing

### Reproducibility

```python
seed = 42
```

Used for both NumPy and TensorFlow.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Rx-9mw/Sentiment_model.git
cd Sentiment_model
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the GUI application:

```bash
python gui.py
```

The application allows users to:

* Configure model parameters
* Train the model
* Monitor training progress
* Save trained models
* Load existing models
* Evaluate model performance

---

## Example Prediction

### Input

```text
This product exceeded my expectations and works perfectly.
```

### Output

```text
Positive Sentiment
```

---

### Input

```text
Completely useless. It broke after two days.
```

### Output

```text
Negative Sentiment
```

---

## Evaluation Metrics

The model supports evaluation using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Classification Report

---

## Project Structure

```text
Sentiment_model/
│
├── data/
│
├── models/
│
├── src/
│   ├── model_schema.py
│   ├── sentiment_model_trainer.py
│   ├── data_processing.py
│   └── gui.py
│
├── requirements.txt
│
└── README.md
```

---

## Engineering Thesis

This repository was developed as part of an Engineering Thesis.

### Thesis Topic

**Sentiment Analysis of Product Reviews Using Bidirectional LSTM Networks with an Attention Mechanism**

The project investigates the effectiveness of recurrent neural networks in natural language processing tasks and evaluates the impact of attention-based architectures on sentiment classification performance.

---

## Future Improvements

* Transformer-based models (BERT, RoBERTa)
* Explainable AI (XAI)
* REST API deployment
* Docker support
* GPU optimization
* Web-based interface

---

## Authors

**Szymon Pawłowicz**

**Piotr Kurzak**

WSB Merito University

---

## License

This project is published for educational and research purposes.
