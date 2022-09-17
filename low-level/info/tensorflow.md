#### Install
* [Install python](https://www.python.org/downloads/)
* [Install tensorflow](https://www.tensorflow.org/install)
    * `pip install --upgrade pip`
    * `pip install tensorflow`
* Install libraries
    * `pip install pandas`
    * `pip install matplotlib`
* [GPU support (optional)](https://www.tensorflow.org/install/gpu)
    * [Install CUDA](https://developer.nvidia.com/cuda-zone)
    * [Install CUDNN](https://developer.nvidia.com/cudnn)
        * Copy everything to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3`
    * Test GPU
        * Task Manager (ctrl+alt+del) -> Performance -> GPU
        * Add to code: `print("GPUs Available: ", len(tf.config.list_physical_devices('GPU')))`
        * Run model training         
* [IntelliJ IDEA](https://plugins.jetbrains.com/plugin/631-python)
    * Open .py file, agree to install python plugins

#### Code snippets
* Loss functions
    * `SparseCategoricalCrossentropy` - multiple labels (y) with no limit on values
    * `BinaryCrossentropy` - single label (y) with values between 0 and 1
* Minimal sequence
```
class TextAugmentation(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set):
        self.x, self.y = x_set, y_set
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x, self.y
```
