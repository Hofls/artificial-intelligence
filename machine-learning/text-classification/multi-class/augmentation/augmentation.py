import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import re

class TextAugmentation(tf.keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x, self.y

def standardizeData(sentences):
  data = []
  for sentence in sentences:
    formatted = sentence.lower().strip()
    onlyText = re.sub('[^a-z0-9 ]+', '', formatted)
    data.append(onlyText)
  return np.array(data)

def getTrainingData():
  csv = pd.read_csv('training-set.csv', delimiter=';')
  labels = csv.to_numpy()[:, 0]
  questions = csv.to_numpy()[:, 1]
  return {
      "labels": np.asarray(labels).astype('float32'),
      "questions": standardizeData(questions)
  }

def toNumbers(text):
  numbers = []
  for symbol in list(text):
    numbers.append(ord(symbol))
  return np.asarray(numbers).astype('float32')

def eachToNumbers(sentences):
  data = []
  for sentence in sentences:
    data.append(toNumbers(sentence))
  return tf.keras.preprocessing.sequence.pad_sequences(data, padding='post')



rawData = getTrainingData()
convertedData = eachToNumbers(rawData["questions"])
labels = rawData["labels"]

model = tf.keras.models.Sequential([
  tf.keras.layers.Embedding(2000, 8),
  tf.keras.layers.GlobalAveragePooling1D(),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(3) # unique labels count
 ])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
gener = TextAugmentation(convertedData, labels, 12)
history = model.fit(gener, epochs=25, verbose=0)
plt.plot(history.history['loss'])

print(model.predict(convertedData))
