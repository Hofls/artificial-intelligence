How to use:
* Train and save model
    * Download [archive](https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip)
    * Unzip archive into `train-model` folder
    * Run `use_trained_model.py`
    * Model should be trained and saved into `saved_models` folder
* Load and use model
    * Copy saved model from `train-model/saved_models` to `use-trained-model/trained_model` 
    * Save random images of dogs and cats from the internet to `use-trained-model/test_images`
    * Run `use_trained_model.py`
    * Model should identify all `.jpg` images in `test_images` folder

Based on:
* https://www.tensorflow.org/tutorials/images/classification
* https://www.tensorflow.org/tutorials/keras/save_and_load
