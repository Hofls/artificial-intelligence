import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import string
import re


class TextAugmentation(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set):
        self.x, self.y = x_set, y_set

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        augmented = augmentEach(self.x)
        text_as_numbers = eachToNumbers(augmented)
        return text_as_numbers, self.y


def standardizeData(sentences):
    data = []
    for sentence in sentences:
        formatted = sentence.lower().strip()
        only_text = re.sub('[^a-z0-9 ]+', '', formatted)
        only_big_words = deleteWords(only_text)
        data.append(only_big_words)
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


def augmentEach(sentences):
    augmented = []
    for sentence in sentences:
        augmented.append(augment(sentence))
    return augmented


def deleteRandomWord(sentence):
    words = sentence.split()
    words.remove(random.choice(words))
    return ' '.join(words)


def deleteWords(sentence):
    return ' '.join(word for word in sentence.split() if len(word) > 2)


def changeWordsOrder(sentence):
    words = sentence.split()
    random.shuffle(words)
    return ' '.join(words)


def addRandomWord(sentence):
    word_length = random.randrange(10) + 3
    new_word = ''.join(random.choices(string.ascii_lowercase, k=word_length))
    return sentence + ' ' + new_word


def augment(sentence):
    # sentence = deleteRandomWord(sentence)
    sentence = addRandomWord(sentence)
    sentence = changeWordsOrder(sentence)
    return sentence


model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(2000, 8),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3)  # unique labels count
])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
rawData = getTrainingData()
textGenerator = TextAugmentation(rawData["questions"], rawData["labels"])
history = model.fit(textGenerator, epochs=80, verbose=0)
plt.plot(history.history['loss'])
plt.show()

print(rawData["questions"])
print(augmentEach(rawData["questions"]))
augmentedData = textGenerator.__getitem__(0)
print(model.predict(augmentedData))
