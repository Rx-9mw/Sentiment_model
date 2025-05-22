import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, BatchNormalization


#-----------------------------------------------------------------
#
#
# I wrote more comments than I realistically should,
# buuuut I thought we could use this as notes for learning and better understanding how this stuff works.
# So yes, most of this file is green and ugly, but at least we can study from it!
#
#
#-----------------------------------------------------------------

# IMPORT DATA FROM OUR CSV FILE
df = pd.read_csv('train.csv', header=None, names=['label','title','review'])

# ASSIGN THE ACTUAL REVIEW AND LABEL TO CORRECT VARIABLES, IN OUR MODEL WE DONT USE TITLES
texts = df['review'].values
labels = df['label'].values

# REDO THE LABELS FROM 1-2 TO 0-1 FOR THE MODEL
labels = labels - 1

# SPECIFY THE AMOUNT OF CLASSES, IN OUR MODEL WE CAN USE THE LEN OF UNIQUE LABELS
num_classes = len(np.unique(labels))

# CATEGORIZE THE LABELS USING THE ONE-HOT ENCODING FOR THE LSTM MODEL
labels_one_hot = to_categorical(labels, num_classes=num_classes)
 
# CREATE A TOKENIZER OBJECT, KEEP 1000 MOST COMMON WORDS, IF WORD NOT SEEN IN TRAINING SET IT TO OOV (OUT-OF-VOCABULARY)
tokenizer = Tokenizer(num_words=1000, oov_token="<DOV>")

# FIT THE TOKENIZER ONTO THE REVIEWS, IT LEARNS WHICH WORDS ARE THE MOST COMMON AND ASSIGNS THEM AN INDEX
tokenizer.fit_on_texts(texts)

# CONVERTS ALL REVIEWS INTO LISTS OF INTEGERS EG. "It was great" -> [14, 2, 45]
sequences = tokenizer.texts_to_sequences(texts)

# PADS OUT ALL SEQUENCES TO BE OF THE SAME LENGTH EG. [14, 2, 45], [12, 13] -> [14, 2, 45], [0, 12, 13] (IF MAXLEN=3)
# MIGHT NEED TO MAKE MAXLEN BIGGER TO ACCOMODATE MORE CONTENT FROM THE REVIEWS
padded = pad_sequences(sequences, maxlen=100)

# SPLITS THE DATASET INTO THE PADDED SEQUENCES (X) AND THE LABELS (Y), AT THIS POINT WE USE 20% OF OUR TRAINING DATA AS TESTING DATA,
# WE HAVE A SEPERATE TESTING FILE SO WE CAN CHANGE THAT LATER
X_train, X_val, Y_train, Y_val = train_test_split(padded, labels_one_hot, test_size=0.2, random_state=42)

# DECLARATION FOR THE ACTUAL LSTM MODEL, WE CAN TWEEK THESE VALUES ON THE FLY IF WE DECIDE TO MAKE A MORE ACCURATE MODEL
# FOR NOW LETS LEAVE IT LIKE IT IS
model = Sequential([
    Embedding(input_dim=1000, output_dim=32),
    LSTM(64, return_sequences=False),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

# "BEST" ACCURACY MODEL ARCHITECTURE FOR LATER, IT USES A LOT OF THINGS THAT I DONT REALLY UNDERSTAND YET SO LETS PUT THAT AWAY FOR NOW
# model = Sequential([
#     Embedding(input_dim=50000, output_dim=300, input_length=max_len),
#     Bidirectional(LSTM(256, return_sequences=True)),
#     Dropout(0.5),
#     Bidirectional(LSTM(128)),
#     Dropout(0.5),
#     Dense(128, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.5),
#     Dense(64, activation='relu'),
#     Dense(2, activation='softmax')
# ])

# NOT SURE IF WE NEED THIS LINE, HAVE TO LOOK INTO IT LATER
model.build(input_shape=(None, padded.shape[1]))

# THIS PRINTS THE SUMMARY OF THE BUILT AND DECLARED MODEL. LAYERS, SHAPES, NUMBER OF PARAMETERS
model.summary()

# CONFIGURATION FOR THE LOSS FUNCTION, THE OPTIMIZER AND THE METRICS, I ADDED ACCURACY CAUSE ITS EASIER TO JUDGE IF EVERYTHING WORKS
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# FITTING THE MODEL WITH THE DATA, RUNNING THROUGH ALL THE PARAMS (EPOCH) 5 TIMES, VALIDATION WITH THE VALIDATION DATA CREATED BEFORE
model.fit(X_train, Y_train, epochs=5, validation_data=(X_val, Y_val))

# AFTER TRAINING, SAVES THE MODEL TO A .keras FILE FOR LATER USAGE
model.save("sentiment_please_dont_brick_my_pc_model.keras")
