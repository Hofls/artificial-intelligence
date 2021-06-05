import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import augment


def print_results(raw_data, text_generator, model, history):
    plt.plot(history.history['loss'])
    plt.show()

    print("Input data:")
    print(raw_data["questions"])
    print(augment.augment_each(raw_data["questions"]))

    print("Predictions (raw):")
    augmentedData = text_generator.__getitem__(0)
    predictions = model.predict(augmentedData)
    print(predictions)

    print("Predictions (normalized):")
    for i in range(len(predictions)):
        print_prediction(predictions[i], raw_data["questions"][i])


def print_prediction(prediction, text):
    score = tf.nn.softmax(prediction)
    print("Label - {}, Confidence - {:.2f}, Text - {}"
          .format(np.argmax(score), 100 * np.max(score), text))


def save_model(model):
    current_directory = os.path.dirname(os.path.realpath(__file__))
    path_to_model = os.path.join(current_directory, "exported-model")
    model.save(path_to_model, save_format='tf')
