# Sentiment Analysis with BiLSTM and Attention Mechanism

## Overview

This project implements a sentiment analysis system for product reviews using a Bidirectional Long Short-Term Memory (BiLSTM) neural network enhanced with a custom Attention mechanism. The solution was developed as part of an engineering thesis focused on natural language processing (NLP) and deep learning techniques.

The application allows users to train, evaluate, and monitor sentiment classification models through a graphical desktop interface built with Tkinter.

## Features

* Binary sentiment classification (Positive / Negative)
* Bidirectional LSTM architecture
* Custom Attention Layer implementation
* Automated text preprocessing pipeline
* Dataset balancing using class weights
* Real-time training monitoring
* Training management through graphical interface
* Automatic learning rate reduction
* Early stopping mechanism
* Model serialization and persistence
* Performance visualization and evaluation

---

## Architecture

The neural network consists of the following layers:

```text
Input Layer
    ↓
Embedding Layer
    ↓
Bidirectional LSTM
    ↓
Attention Layer
    ↓
Dense Layer (ReLU + L2 Regularization)
    ↓
Dropout
    ↓
Output Layer (Softmax)
```

### Model Parameters

| Component           | Configuration            |
| ------------------- | ------------------------ |
| Embedding Dimension | 150                      |
| LSTM Units          | 16                       |
| Dense Units         | 32                       |
| Recurrent Dropout   | 0.2                      |
| Dropout             | 0.5                      |
| Output Classes      | 2                        |
| Activation Function | Softmax                  |
| Optimizer           | Adam                     |
| Loss Function       | Categorical Crossentropy |

---

## Technologies Used

### Core Technologies

* Python 3.9
* TensorFlow
* Keras
* NumPy
* Pandas
* Scikit-Learn
* NLTK
* Tkinter

### Additional Components

* Custom Attention Layer
* Thread-based training execution
* Dataset preprocessing pipeline
* Visualization utilities

---

## Dataset

The model was trained using a subset of the Amazon Reviews Dataset.

Dataset characteristics:

* Product reviews in English
* Approximately 3.2 million records used for training
* Binary sentiment labels
* CSV format

The preprocessing pipeline includes:

* Lowercase normalization
* HTML removal
* URL removal
* Contraction expansion
* Non-ASCII character filtering
* Stopword removal
* Tokenization
* Vectorization

---

## Training Strategy

The project utilizes several techniques to improve model generalization:

### Regularization

* L2 Weight Regularization
* Dropout (0.5)
* Recurrent Dropout (0.2)

### Optimization

* Adam Optimizer
* ReduceLROnPlateau
* Early Stopping
* Class Weight Balancing

### Reproducibility

A fixed random seed (`42`) is used for TensorFlow and NumPy to ensure reproducible experiments.

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/sentiment-analysis-lstm.git

cd sentiment-analysis-lstm
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Project Structure

```text
project/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── models/
│   └── saved_models/
│
├── src/
│   ├── model_schema.py
│   ├── data_processing.py
│   ├── sentiment_model_trainer.py
│   ├── attention_layer.py
│   └── gui.py
│
├── visualizations/
│
├── requirements.txt
│
└── README.md
```

---

## Running the Application

Launch the graphical interface:

```bash
python gui.py
```

The GUI allows users to:

* Configure model parameters
* Start and stop training
* Monitor progress in real time
* Save trained models
* Generate performance charts

---

## Evaluation Metrics

The model is evaluated using:

* Accuracy
* Precision
* Recall
* Classification Report
* Confusion Matrix

---

## Future Improvements

Potential enhancements include:

* Transformer-based architectures (BERT, RoBERTa)
* Multi-class sentiment classification
* Aspect-Based Sentiment Analysis
* GPU acceleration support
* REST API deployment
* Docker containerization
* Web application interface

---

## Authors

**Szymon Pawłowicz**

**Piotr Kurzak**

Engineering Thesis
Faculty of Computer Science
WSB Merito University

---

## License

This project is provided for educational and research purposes.
