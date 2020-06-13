import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

import time
from collections import namedtuple


def showImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def getDirectoriesStats(train_dir, validation_dir):
    train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
    train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
    validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures

    num_cats_tr = len(os.listdir(train_cats_dir))
    num_dogs_tr = len(os.listdir(train_dogs_dir))

    num_cats_val = len(os.listdir(validation_cats_dir))
    num_dogs_val = len(os.listdir(validation_dogs_dir))

    total_train = num_cats_tr + num_dogs_tr
    total_val = num_cats_val + num_dogs_val

    print('total training cat images:', num_cats_tr)
    print('total training dog images:', num_dogs_tr)

    print('total validation cat images:', num_cats_val)
    print('total validation dog images:', num_dogs_val)
    print("--")
    print("Total training images:", total_train)
    print("Total validation images:", total_val)

    DirTotal = namedtuple('DirTotal', 'total_train total_val')
    dirTotal = DirTotal(total_train, total_val)
    return dirTotal


def trainModel(train_dir, validation_dir, dirTotal):
    IMG_HEIGHT = 150
    IMG_WIDTH = 150

    train_image_generator = ImageDataGenerator(rescale=1./255,
                        rotation_range=45,
                        width_shift_range=.15,
                        height_shift_range=.15,
                        horizontal_flip=True,
                        zoom_range=0.5) # Generator for our training data
    validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

    train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                               directory=train_dir,
                                                               shuffle=True,
                                                               target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                               class_mode='binary')

    val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                                  directory=validation_dir,
                                                                  target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                  class_mode='binary')

    sample_training_images, _ = next(train_data_gen)


    augmented_images = [train_data_gen[0][0][0] for i in range(5)]
    showImages(augmented_images)
    #showImages(sample_training_images[:5])


    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    history = model.fit(
        train_data_gen,
        steps_per_epoch=dirTotal.total_train // batch_size,
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps=dirTotal.total_val // batch_size
    )
    ModelInfo = namedtuple('ModelInfo', 'model history')
    modelInfo = ModelInfo(model, history)
    return modelInfo


def showHistoryOnGraphs(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss=history.history['loss']
    val_loss=history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def saveModel(model):
    t = time.time()
    saved_models_directory = os.path.join(current_directory, "saved_models")
    export_path = os.path.join(saved_models_directory, "{}".format(int(t)))
    model.save(export_path, save_format='tf')

batch_size = 16 #todo - change back to 125
epochs = 3 #todo - change back to 16

current_directory = os.path.dirname(os.path.realpath(__file__))
images_directory = os.path.join(current_directory, 'cats_and_dogs_filtered')

train_dir = os.path.join(images_directory, 'train')
validation_dir = os.path.join(images_directory, 'validation')

dirTotal = getDirectoriesStats(train_dir, validation_dir)

modelInfo = trainModel(train_dir, validation_dir, dirTotal)

showHistoryOnGraphs(modelInfo.history)

saveModel(modelInfo.model)

