import bz2
import numpy as np
from tensorflow.python.keras import models, layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Flatten
from tensorflow.keras import regularizers

def get_labels_and_texts(file, n=10000):
    labels = []
    texts = []
    i = 0
    for line in bz2.BZ2File(file):
        x = line.decode("utf-8")
        labels.append(int(x[9]) - 1)
        texts.append(x[10:].strip())
        i = i + 1
        if i >= n:
          return np.array(labels), texts
    return np.array(labels), texts

train_labels, train_texts = get_labels_and_texts('data/train.ft.txt.bz2')

# Initialize the tokenizer with a specific vocabulary size
tokenizer = Tokenizer(num_words=5000)

# Fit the tokenizer on your text data
tokenizer.fit_on_texts(train_texts)

# Convert the text data to numerical data
train_sequences = tokenizer.texts_to_sequences(train_texts)

# Pad the sequences so they're all the same length
MAX_LENGTH = 200  # Or whatever length is appropriate for your data
train_padded = pad_sequences(train_sequences, maxlen=MAX_LENGTH)

# Define the size of your vocabulary and the dimension of your embeddings
VOCAB_SIZE = 5000
EMBEDDING_DIM = 50

ffnn = models.Sequential()
ffnn.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH))
ffnn.add(Flatten())
ffnn.add(layers.Dense(100, activation="sigmoid"))
ffnn.add(layers.Dropout(0.5))
ffnn.add(layers.Dense(50, activation="sigmoid"))
ffnn.add(layers.Dropout(0.5))
ffnn.add(layers.Dense(1, activation="sigmoid"))

ffnn.build(input_shape=(None, MAX_LENGTH))
ffnn.summary()

ffnn.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])

ffnn.fit(train_padded, train_labels, epochs=10, batch_size=10, verbose=1)
