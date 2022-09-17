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
        augmented = augment_each(self.x)
        text_as_numbers = each_to_numbers(augmented)
        return text_as_numbers, self.y


def standardize_data(sentences):
    data = []
    for sentence in sentences:
        formatted = sentence.lower().strip()
        only_text = re.sub('[^a-z0-9 ]+', '', formatted)
        only_big_words = delete_words(only_text)
        data.append(only_big_words)
    return np.array(data)


def get_training_data():
    csv = pd.read_csv('training-set.csv', delimiter=';')
    labels = csv.to_numpy()[:, 0]
    questions = csv.to_numpy()[:, 1]
    return {
        "labels": np.asarray(labels).astype('float32'),
        "questions": standardize_data(questions)
    }


def to_numbers(text):
    numbers = []
    for symbol in list(text):
        numbers.append(ord(symbol))
    return np.asarray(numbers).astype('float32')


def each_to_numbers(sentences):
    data = []
    for sentence in sentences:
        data.append(to_numbers(sentence))
    return tf.keras.preprocessing.sequence.pad_sequences(data, padding='post')


def augment_each(sentences):
    augmented = []
    for sentence in sentences:
        augmented.append(augment(sentence))
    return augmented


def delete_random_word(sentence):
    words = sentence.split()
    words.remove(random.choice(words))
    return ' '.join(words)


def delete_words(sentence):
    return ' '.join(word for word in sentence.split() if len(word) > 2)


def change_words_order(sentence):
    words = sentence.split()
    random.shuffle(words)
    return ' '.join(words)


def add_random_word(sentence):
    word_length = random.randrange(10) + 3
    new_word = ''.join(random.choices(string.ascii_lowercase, k=word_length))
    return sentence + ' ' + new_word


def augment(sentence):
    # sentence = deleteRandomWord(sentence)
    sentence = add_random_word(sentence)
    sentence = change_words_order(sentence)
    return sentence


model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(200, 128),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3)  # unique labels count
])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
rawData = get_training_data()
textGenerator = TextAugmentation(rawData["questions"], rawData["labels"])
history = model.fit(textGenerator, epochs=80, verbose=0)
plt.plot(history.history['loss'])
plt.show()

print(rawData["questions"])
print(augment_each(rawData["questions"]))
augmentedData = textGenerator.__getitem__(0)
print(model.predict(augmentedData))
