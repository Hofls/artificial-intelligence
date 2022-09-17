import string
import random
import numpy as np
import tensorflow as tf


def augment(sentence):
    #sentence = delete_random_word(sentence)
    sentence = add_random_words(sentence)
    sentence = change_words_order(sentence)
    return sentence


class TextAugmentation(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set):
        self.x, self.y = x_set, y_set

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        augmented = augment_each(self.x)
        text_as_numbers = each_to_numbers(augmented)
        return text_as_numbers, self.y


def change_words_order(sentence):
    words = sentence.split()
    random.shuffle(words)
    return ' '.join(words)


def add_random_words(sentence):
    words = sentence.split()
    new_words_count = len(words) // 3
    word_length = random.randrange(10) + 3
    for i in range(new_words_count):
        new_word = ''.join(random.choices(string.ascii_lowercase, k=word_length))
        sentence = sentence + ' ' + new_word
    return sentence


def delete_random_word(sentence):
    words = sentence.split()
    words.remove(random.choice(words))
    return ' '.join(words)


def augment_each(sentences):
    augmented = []
    for sentence in sentences:
        augmented.append(augment(sentence))
    return augmented


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
