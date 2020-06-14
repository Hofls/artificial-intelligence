# Expectations:
# This folder contains subfolder called 'test_images', with bunch of images
# This folder contains subfolder called 'trained_model', with model exported in SavedModel format (contains assets, variables, saved_model.pb)

import tensorflow as tf
import glob, os
import numpy as np
from PIL import Image
import pickle

def identifyPicture(pic, loaded_model):
    pic = pic.resize((150, 150), Image.ANTIALIAS)
    pix = np.array(pic)

    result = loaded_model.predict(pix[np.newaxis, ...])
    print(file + " - " + class_names[np.argmax(result)])

current_directory = os.path.dirname(os.path.realpath(__file__))
images_directory = os.path.join(current_directory, "test_images")
model_directory = os.path.join(current_directory, "trained_model")
custom_classes_path = os.path.join(model_directory, "custom_classes")

with open (custom_classes_path, 'rb') as fp:
    class_names = pickle.load(fp)
loaded_model = tf.keras.models.load_model(model_directory)
#loaded_model.summary() # Check its architecture

os.chdir(images_directory)
for file in glob.glob("*.*"):
    pic = Image.open(file)
    identifyPicture(pic, loaded_model)
