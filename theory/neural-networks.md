## Concepts
* `Artificial intelligence` -  intelligence demonstrated by machines
* `Machine learning` - systems that improve/learn automatically
* `Neural network` - systems inspired by the biological neural networks that constitute animal brains
    * Consist of `nodes` (neurons) connected to each other via `edges` (synapses) 
    * Each node/edge has `weight` that increases/decreases strength of signal
    * Each node takes multiple numbers as input (`signal`) and produces single number as output (via `Activation function`)
    * Nodes aggregated into `layers` (different layers perform different transformations). Typical layers: input - hidden - output
    * Networks `learn` by processing examples, making guesses and adjusting weights
* `Tensor` - multi-dimensional array of numbers
* `Hyperparameter` - parameters to control learning process (Hidden layers, Activation functions, Dropout, Learning rate)
* `Backpropagation` - how much each node contributed to the error (to adjust their weights)
* 

## Types
* `GAN` (Generative Adversarial Network) - 
* `Deep learning` - multiple layers in the network to progressively extract higher-level features (edges => nose, eyes => face)
    * `CNN` (Convolutional NN) - inspired by animal visual cortex. Each neuron in one layer connected to all neurons in the next layer
        * Usage - recognition, classification, recommendation, language processing
    * `RNN` (Recurrent  NN) - connections between nodes form a directed graph along a temporal sequence
        * Usage - modeling sequence data (time series, natural language..)
    * `DBN` (Deep Belief Network) - each layer acts as a feature detector
        * Usage - generation, recognition
* `Autoencoder` - encodes data by ignoring noise; reconstructs original data from encoded one
        
## Misc
* Frameworks
    * By abstraction level: `Keras` > `Tensorflow` > `Pythorch`
* `Deepfake` - media in which a person replaced with someone else
* `GPT-3` (Generative Pre-trained Transformer 3) - state-of-the-art language model (year 2020)
    * Produces human-like text (poetry, books, computer code etc)
    * [Turing test](https://lacker.io/ai/2020/07/06/giving-gpt-3-a-turing-test.html)
