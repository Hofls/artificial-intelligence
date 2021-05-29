#### Concepts
* `Artificial intelligence` -  intelligence demonstrated by machines
* `Machine learning` - systems that improve/learn automatically
* `Neural network` - systems inspired by the biological neural networks that constitute animal brains
    * Consist of `nodes` (neurons) connected to each other via `edges` (synapses) 
    * Each node/edge has `weight` that increases/decreases strength of signal
    * Each node takes multiple numbers as input (`signal`) and produces single number as output (via `Activation function`)
    * Nodes aggregated into `layers` (different layers perform different transformations). Typical layers: input - hidden - output
    * Networks `learn` by processing examples, making guesses and adjusting weights
* `Tensor` - multi-dimensional array of numbers
* `Hyperparameter` - parameters to control learning process 
    * Hidden layers, Activation functions, Dropout, Learning rate
* `Backpropagation` - how much each node contributed to the error (to adjust their weights)
* `Feature` - input variable (measurable property)
    * In spam detection - email headers, email structure, language, frequency of specific terms, grammatical correctness
* `Label` - thing we are predicting (e.g. price of bread, what's in the picture)
* `Labeled example` = features + label (used for training)
    * Email with mark "spam/not spam"
* `Inference` - applying trained model to unlabeled examples  
* `Preprocess` - get data ready (clean, transform, reduce)
    * In image recognition - remove noise, scale down, convert to black & white
* `Loss` - how bad is model prediction on specific example?
    * e.g. 0 - prediction is perfect, 0.05 - pretty close, 0.9 - not even close
* `Loss function` - adjusts weights based on losses
    * e.g. 0 - weights stays the same, 0.05 - adjusts a little, 0.9 adjusts a lot
* 

#### Network types
* `GAN` (Generative Adversarial Network) - 
* `RNN` (Recurrent neural network) - 
* `CNN` (Convolutional neural network) - 
* 
* `Deep learning` - multiple layers in the network to progressively extract higher-level features (edges => nose, eyes => face)
    * `CNN` (Convolutional NN) - inspired by animal visual cortex. Each neuron in one layer connected to all neurons in the next layer
        * Usage - recognition, classification, recommendation, language processing
    * `RNN` (Recurrent  NN) - connections between nodes form a directed graph along a temporal sequence
        * Usage - modeling sequence data (time series, natural language..)
    * `DBN` (Deep Belief Network) - each layer acts as a feature detector
        * Usage - generation, recognition
* `Autoencoder` - encodes data by ignoring noise; reconstructs original data from encoded one

#### Approaches
* `Supervised learning` - training on labeled examples, goal is to learn rules that maps inputs to outputs
* `Unsupervised learning` - no labels, used to discover hidden patterns/feature learning
* `Reinforcement learning` - interaction with dynamic environment (e.g. move car from A to B)
    * With constant feedback (analogous to rewards) that it tries to optimize

#### Problem types
* `Analytics` - extract information from text (e.g. What is the capital of India?)
* `Regression` - predict values (e.g. What this stock price will be tomorrow?)
* `Classification` - classify objects (e.g. what this image represents?)
* `Recognition` - (e.g. what's written here?)
* `Detection`
* `Generation`
* `Recommenders` - create recommendations (e.g. what movies this person would like?)
* `Clustering` - discover structure (e.g. recognize communities within large group of people)
* `Anomaly detection` - find unusual occurrences (e.g. is something suspicious happened?)

#### Areas
* `Text`
    * 
        
#### Misc
* Frameworks
    * By abstraction level: `Keras` > `Tensorflow` > `Pytorch`
* `Kaggle.com` - competitions, datasets, notebooks
* `Deepfake` - media in which a person replaced with someone else
* `GPT-3` (Generative Pre-trained Transformer 3) - state-of-the-art language model (year 2020)
    * Produces human-like text (poetry, books, computer code etc)
    * [Turing test](https://lacker.io/ai/2020/07/06/giving-gpt-3-a-turing-test.html)
