# Expectations:
# Model was trained to identify 150x150 images of cats and dogs
# This folder contains subfolder called 'test_images', with bunch of .jpg images
# This folder contains subfolder called 'trained_model', with model exported in SavedModel format (contains assets, variables, saved_model.pb)

import tensorflow as tf
import glob, os
import numpy as np
from PIL import Image

def identifyPicture(pic, loaded_model):
    pic = pic.resize((150, 150), Image.ANTIALIAS)
    #print(pic)
    pix = np.array(pic)
    #print(pix.shape)
    #print(pix[np.newaxis, ...].shape)

    result = loaded_model.predict(pix[np.newaxis, ...])
    if (result > 0): # Result depends on activation function that is used during model training
        print(file + ' is a dog')
    else:
        print(file + ' is a cat')

current_directory = os.path.dirname(os.path.realpath(__file__))
images_directory = os.path.join(current_directory, "test_images")
model_directory = os.path.join(current_directory, "trained_model")

loaded_model = tf.keras.models.load_model(model_directory)
#loaded_model.summary() # Check its architecture

os.chdir(images_directory)
for file in glob.glob("*.jpg"):
    pic = Image.open(file)
    identifyPicture(pic, loaded_model)
