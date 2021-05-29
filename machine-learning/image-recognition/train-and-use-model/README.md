### How to run:
* Train and save model
    * Download [archive](https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip)
    * Unzip archive into `train-model/images` folder
    * Run `train_model.py`
    * Model should be trained and saved into `saved_models` folder
    
* Load and use model
    * Copy saved model from `train-model/saved_models` to `use-trained-model/trained_model` 
    * Save random images of dogs and cats from the internet to `use-trained-model/test_images`
    * Run `use_trained_model.py`
    * Model should identify all images in `test_images` folder
    
### Misc
* Continue training of existing model
    * Copy model from `train-model/saved_models` to `train-model/model_to_train`
    * Run `train_model.py`
    
* Train for a long time with periodical backups:
    * Set `endless_mode = True` in `train_model.py`
    * When you are done training - press `ctrl+c`
    
* Find images that model is unable to identify:
    * Copy saved model from `train-model/saved_models` to `find-model-problems/trained_model` 
    * Copy images from `train-model/images/validation` to `find-model-problems/validation_images`
    * Run `find_model_problems.py`
    * All problematic images will appear in `find-model-problems/unidentified` directories

### Based on:
* https://www.tensorflow.org/tutorials/keras/classification
* https://www.tensorflow.org/tutorials/images/classification
* https://www.tensorflow.org/tutorials/keras/save_and_load
