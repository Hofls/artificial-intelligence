# Expectations:
# This folder contains subfolders 'train/cats' and `train/dogs` with training images
# This folder contains subfolders 'validation/cats' and `validation/dogs` with validation images

# Model:
# If there is a model inside 'model_to_train' folder - it is used for further training
# If no model is found - new one is automatically created

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

import time
from collections import namedtuple
import pickle


def showImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Understand the data
def getDirectoriesStats(train_dir, validation_dir):
    total_train = sum([len(files) for r, d, files in os.walk(train_dir)])
    total_val = sum([len(files) for r, d, files in os.walk(validation_dir)])

    print("Total training images:", total_train)
    print("Total validation images:", total_val)

    DirTotal = namedtuple('DirTotal', 'total_train total_val')
    dirTotal = DirTotal(total_train, total_val)
    return dirTotal

def createNewModel(IMG_HEIGHT, IMG_WIDTH, train_dir):
    os.chdir(train_dir)
    folders_count = len([name for name in os.listdir('.') if not os.path.isfile(name)])

    # Create the model
    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu',
               input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(folders_count) # Amount of elements in prediction array [dog_score, cat_score, bird_score]
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

def loadModel():
    current_directory = os.path.dirname(os.path.realpath(__file__))
    model_directory = os.path.join(current_directory, "model_to_train")
    model_file = os.path.join(model_directory, "saved_model.pb")
    if os.path.isfile(model_file):
        return tf.keras.models.load_model(model_directory)
    else:
        return None

def trainModel(train_dir, validation_dir, dirTotal):
    # Data augmentation (the model will never see the exact same picture twice during training)
    IMG_HEIGHT = 150
    IMG_WIDTH = 150

    train_image_generator = ImageDataGenerator(rescale=1./255,
                        rotation_range=25,
                        width_shift_range=.15,
                        height_shift_range=.15,
                        horizontal_flip=True,
                        shear_range=0.2,
                        brightness_range=(0.75, 1.25),
                        channel_shift_range=50.0,
                        zoom_range=0.2) # Generator for our training data
    validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

    train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                               directory=train_dir,
                                                               shuffle=True,
                                                               target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                               class_mode='categorical')

    val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                                  directory=validation_dir,
                                                                  target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                  class_mode='categorical')

    sample_training_images, _ = next(train_data_gen)


    augmented_images = [train_data_gen[0][0][0] for i in range(5)]
    showImages(augmented_images)

    # Get the model
    model = loadModel()
    if model is None:
        model = createNewModel(IMG_HEIGHT, IMG_WIDTH, train_dir)

    model.summary()

    histories = []
    if (endless_mode):
        try:
            for x in range(60):
                history = model.fit(
                    train_data_gen,
                    steps_per_epoch=dirTotal.total_train // batch_size,
                    epochs=epochs,
                    validation_data=val_data_gen,
                    validation_steps=dirTotal.total_val // batch_size
                )
                saveModel(model, train_dir)
                histories.append(history)
        except KeyboardInterrupt:
            pass
    else:
        history = model.fit(
            train_data_gen,
            steps_per_epoch=dirTotal.total_train // batch_size,
            epochs=epochs,
            validation_data=val_data_gen,
            validation_steps=dirTotal.total_val // batch_size
        )
        histories.append(history)
    ModelInfo = namedtuple('ModelInfo', 'model histories')
    modelInfo = ModelInfo(model, histories)
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


def saveModel(model, train_dir):
    t = time.time()
    saved_models_directory = os.path.join(current_directory, "saved_models")
    export_path = os.path.join(saved_models_directory, "{}".format(int(t)))
    model.save(export_path, save_format='tf')

    dir_list = next(os.walk(train_dir))[1]
    custom_classes_path = os.path.join(export_path, "custom_classes")
    with open(custom_classes_path, 'wb') as fp:
        pickle.dump(dir_list, fp)

batch_size = 125 #todo - change back to 125
epochs = 10 #todo - change back to 16
endless_mode = False

current_directory = os.path.dirname(os.path.realpath(__file__))
images_directory = os.path.join(current_directory, 'images')

train_dir = os.path.join(images_directory, 'train')
validation_dir = os.path.join(images_directory, 'validation')

dirTotal = getDirectoriesStats(train_dir, validation_dir)

modelInfo = trainModel(train_dir, validation_dir, dirTotal, endless_mode)

for history in modelInfo.histories:
    showHistoryOnGraphs(history)

saveModel(modelInfo.model, train_dir)

