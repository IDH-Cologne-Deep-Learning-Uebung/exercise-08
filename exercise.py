import bz2

import numpy as np

#from tensorflow.python.keras import models, layers
# from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras import regularizers
import keras
import tensorflow as tf
from keras.preprocessing.text import Tokenizer, text_to_word_sequence   #version Problem
from keras.preprocessing.sequence import pad_sequences    #version Problem
from keras import layers
from keras import regularizers

#Versionsprobleme irgendwie damit behoben


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


train_labels, train_texts = get_labels_and_texts('exercise-08/data/train.ft.txt.bz2', 60000)
#n verändern?...


#preprocess
#train_texts to integer array ...

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts)
vocab_size = len(tokenizer.word_index) + 1
train_texts = tokenizer.texts_to_sequences(train_texts)

MAX_LENGTH = max(len(train_ex) for train_ex in train_texts)

train_texts = pad_sequences(train_texts , maxlen=MAX_LENGTH , padding="post")


ffnn = tf.keras.models.Sequential()
ffnn.add(layers.Input(shape=(MAX_LENGTH)))
ffnn.add(layers.Embedding(vocab_size , 200, input_length=MAX_LENGTH))
ffnn.add(layers.Flatten())
ffnn.add(layers.Dense(100, activation="sigmoid", activity_regularizer=regularizers.L2(0.4)))
ffnn.add(layers.Dropout(0.5))
ffnn.add(layers.Dense(50, activation="sigmoid"))
ffnn.add(layers.Dropout(0.5))
ffnn.add(layers.Dense(1, activation="sigmoid"))

ffnn.summary()

ffnn.compile(loss="binary_crossentropy", optimizer="sgd",
  metrics=["accuracy"])

ffnn.fit(train_texts, train_labels, epochs=10, batch_size=10, verbose=1)


# 200 (n=10000) -> 0,5084 (l=0,6967)
# 300 -> 0,5050 ; +longer (l=0,6972)

#Data Set
# 200 und n= 10000 -> 0,5084 (l=0,6967)
# 200 und n= 13000 -> 0,5030 (l=0,6972)
# 200 und n= 30000 -> 0,5034 (l=0,6954)
# 200 und n= 60000 -> 0,5065 (l=0,6945)