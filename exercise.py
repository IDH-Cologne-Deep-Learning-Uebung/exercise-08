import bz2

import numpy as np

from tensorflow.python.keras import models, layers
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers


def get_labels_and_texts(file, n=10000): #n ist die anzahl der gelesenen Zeilen
    labels = []
    texts = []
    i = 0
    for line in bz2.BZ2File(file): # für jede Zeile im Text...
        x = line.decode("utf-8") ## ...führt Python diesen Code aus
        labels.append(int(x[9]) - 1)
        texts.append(x[10:].strip())
        i = i + 1
        if i >= n:
          return np.array(labels), texts
    return np.array(labels), texts


train_labels, train_texts = get_labels_and_texts('data/train.ft.txt.bz2') # Trainingslabels und Trainingstext werden aus der train.ft.txt.bz2-Datei extrahiert

## Step 3:  Tokenizer
# Tokenizerklasse bereits importiert pad_sequences auch


tokenizer = Tokenizer() 
tokenizer.fit_on_texts(train_texts) #Tokenizer auf trainingsdaten anwenden


vocab_size = len(tokenizer.word_index) + 1 #(+1 weil der Index sonst bei 0 anfängt)
train_sequences = tokenizer.texts_to_sequences(train_texts) # Text in Integer-Arrays umwandeln

MAX_LENGTH = max(len(train_ex) for train_ex in train_texts)
padded_train_sequences = pad_sequences(train_sequences, maxlen=MAX_LENGTH)#Padding festlegen, es wird auf die Länge des längsten Textes festgelegt

#train_texts = pad_sequences(train_texts, maxlen=MAX_LENGTH, padding="post")



## Modell mit Embedding erstellen
# Model wird erstellt
ffnn = models.Sequential()

# Embedding-Schicht hinzufügen
EMBEDDING_DIM = 50  # Beispielwert für die Dimension der Embeddings, du kannst dies anpassen
ffnn.add(layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=EMBEDDING_DIM, input_length=MAX_LENGTH))
ffnn.add(layers.Flatten())  # Flatten-Schicht hinzufügen
# weitere Schichten
ffnn.add(layers.Input(shape=(MAX_LENGTH)))
ffnn.add(layers.Dense(100, activation="sigmoid"))
ffnn.add(layers.Dropout(0.5))
ffnn.add(layers.Dense(50, activation="sigmoid"))
ffnn.add(layers.Dropout(0.5))
ffnn.add(layers.Dense(1, activation="sigmoid"))

#Model wird zusammengefasst
ffnn.summary()

# Model kompilieren 
ffnn.compile(loss="binary_crossentropy", optimizer="sgd",
  metrics=["accuracy"])

## Training beginnt
ffnn.fit(train_texts, train_labels, epochs=10, batch_size=10, verbose=1)



# Zugriff auf die gespeicherten Metriken
accuracy = history.history['accuracy']
loss = history.history['loss']

# Beispiel für Berechnung des Recall (kann je nach Anwendung variieren)
# Beachte: Du musst die Vorhersagen des Modells für deine Testdaten erhalten, um den Recall zu berechnen.
# Hier verwende ich einfach die Trainingsdaten als Beispiel.
train_predictions = ffnn.predict(padded_train_sequences)
train_predictions = np.round(train_predictions)  # Umwandlung der Wahrscheinlichkeiten in Binärwerte (0 oder 1)
train_recall = recall_score(train_labels, train_predictions)

print(f'Accuracy: {accuracy}')
print(f'Loss: {loss}')



ffnn.fit(padded_train_sequences, train_labels, epochs=10, batch_size=10, verbose=1)

## Step 4: Embedding
