import bz2

import numpy as np

from tensorflow.python.keras import models, layers
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.preprocessing.text import Tokenizer


def get_labels_and_texts(file, n=100000):
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


train_labels, train_texts = get_labels_and_texts('train.ft.txt.bz2')

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts)

# Convert text to sequences of integers
sequences = tokenizer.texts_to_sequences(train_texts)

# Pad sequences to a fixed length
max_length = 10000  # Set your desired fixed length
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

vocab_size = len(tokenizer.word_index) + 1


ffnn = models.Sequential()
ffnn.add(layers.Input(shape=(max_length)))
ffnn.add(Embedding(input_dim=vocab_size, output_dim=32, input_length=max_length))
ffnn.add(Flatten())
ffnn.add(layers.Dense(100, activation="sigmoid"))
ffnn.add(layers.Dropout(0.5))
ffnn.add(layers.Dense(50, activation="sigmoid"))
ffnn.add(layers.Dropout(0.5))
ffnn.add(layers.Dense(1, activation="sigmoid"))

ffnn.summary()

ffnn.compile(loss="binary_crossentropy", optimizer="sgd",
  metrics=["accuracy"])

ffnn.fit(padded_sequences, train_labels, epochs=10, batch_size=10, verbose=1)


