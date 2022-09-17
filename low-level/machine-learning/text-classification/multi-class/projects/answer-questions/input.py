import pandas as pd
import numpy as np
import tensorflow as tf
import re
import os


def get_training_data():
    csv = pd.read_csv('data/training-set.csv', delimiter=';')
    labels = csv.to_numpy()[:, 0]
    questions = csv.to_numpy()[:, 1]
    return {
        "labels": np.asarray(labels).astype('float32'),
        "questions": standardize_data(questions)
    }

def standardize_data(sentences):
    data = []
    for sentence in sentences:
        formatted = sentence.lower().strip()
        only_text = re.sub('[^a-z0-9 ]+', '', formatted)
        only_big_words = delete_words(only_text)
        data.append(only_big_words)
    return np.array(data)

def delete_words(sentence):
    return ' '.join(word for word in sentence.split() if len(word) > 2)

def load_model():
    current_directory = os.path.dirname(os.path.realpath(__file__))
    path_to_model = os.path.join(current_directory, "exported-model")
    loaded_model = tf.keras.models.load_model(path_to_model)
    return loaded_model