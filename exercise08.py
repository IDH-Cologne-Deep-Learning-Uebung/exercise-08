import bz2

import numpy as np

from tensorflow.python.keras import models, layers
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense


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
MAX_LENGTH = 500 
VOCAB_SIZE = 5000 

tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(train_texts)
sequences = tokenizer.texts_to_sequences(train_texts)
padded = pad_sequences(sequences, maxlen=MAX_LENGTH)

ffnn = models.Sequential()
ffnn.add(layers.Input(shape=(MAX_LENGTH)))
ffnn.add(layers.Dense(100, activation="sigmoid"))
ffnn.add(layers.Dropout(0.5))
ffnn.add(layers.Dense(50, activation="sigmoid"))
ffnn.add(layers.Dropout(0.5))
ffnn.add(layers.Dense(1, activation="sigmoid"))

ffnn.summary()

ffnn.compile(loss="binary_crossentropy", optimizer="sgd",
  metrics=["accuracy"])

ffnn.fit(train_texts, train_labels, epochs=10, batch_size=10, verbose=1)

embedding_dim = 50
vocab_size = len(tokenizer.word_index) + 1

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=MAX_LENGTH))
model.add(Flatten())
model.add(Dense(100, activation="sigmoid"))
model.add(Dropout(0.5))
model.add(Dense(50, activation="sigmoid"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

