# Expectations:
# This folder contains subfolder called 'validation_images' (should contain folders/files copied from `images/validation` folder)
# This folder contains subfolder called 'trained_model', with model exported in SavedModel format (contains assets, variables, saved_model.pb)

# All the unidentified images will be copied to `unidentified` folder

import tensorflow as tf
import glob, os
import numpy as np
from PIL import Image
import pickle
from shutil import copy2
from pathlib import Path

def identifyPicture(pic, loaded_model):
    pic = pic.resize((150, 150), Image.ANTIALIAS)
    pix = np.array(pic)

    result = loaded_model.predict(pix[np.newaxis, ...])
    return class_names[np.argmax(result)]

current_directory = os.path.dirname(os.path.realpath(__file__))
images_directory = os.path.join(current_directory, "validation_images")
model_directory = os.path.join(current_directory, "trained_model")
unidentified_directory = os.path.join(current_directory, "unidentified")
custom_classes_path = os.path.join(model_directory, "custom_classes")

with open (custom_classes_path, 'rb') as fp:
    class_names = pickle.load(fp)
loaded_model = tf.keras.models.load_model(model_directory)
#loaded_model.summary() # Check its architecture

for class_name in class_names:
    class_directory = os.path.join(images_directory, class_name)
    unidentified_class_directory = os.path.join(unidentified_directory, class_name)
    Path(unidentified_class_directory).mkdir(parents=True, exist_ok=True)
    os.chdir(class_directory)
    identified_count = 0.0
    total_count = 0.0
    for file in glob.glob("*.*"):
        total_count += 1
        pic = Image.open(file)
        identified_class = identifyPicture(pic, loaded_model)
        if (class_name == identified_class):
            #print(file + ' - OK');
            identified_count += 1
        else:
            #print(file + ' - FAIL');
            file_path = os.path.join(class_directory, file)
            copy2(file_path, unidentified_class_directory)
    print('Identified ' + class_name + ' - ' + str(identified_count / total_count))
