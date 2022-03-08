#### Concepts
* `Artificial intelligence` -  intelligence demonstrated by machines
* `Machine learning` - systems that improve/learn automatically
    * High quality and quantity of data is the most important component of success in ML
* `Neural network` - systems inspired by the biological neural networks that constitute animal brains
    * Consist of `nodes` (neurons) connected to each other via `edges` (synapses) 
    * Each node/edge has `weight` that increases/decreases strength of signal
    * Each node takes multiple numbers as input (`signal`) and produces single number as output (via `Activation function`)
    * Nodes aggregated into `layers` (different layers perform different transformations). Typical layers: input - hidden - output
    * Networks `learn` by processing examples, making guesses and adjusting weights
* Problem solving algorithm:
    * Define ML problem, propose a solution -> Construct dataset -> Transform data -> Train a model -> Use model to make predictions
* `Tensor` - multi-dimensional array of numbers
* `Hyperparameter` - parameters to control learning process 
    * Hidden layers, Activation functions, Dropout, Learning rate
* `Backpropagation` - how much each node contributed to the error (to adjust their weights)
    * `Gradient descent` - iteratively adjusts weights to minimize loss 
        * If node contributed to error - its weight decreases, if contributed to success - weight increases
    * `Learning rate` - higher rate means higher learning speed and lower accuracy (e.g. 0.1)
    * `Gradient step` = learning rate * gradient
    * `Batch` - set of examples used in one iteration (gradient update)
* `Feature` - input variable (measurable property)
    * In spam detection - email headers, email structure, language, frequency of specific terms, grammatical correctness
* `Feature engineering` - converting data to useful features
    * Helps model understand the data the same way people do
* `Label` - thing we are predicting (e.g. price of bread, what's in the picture)
* `Labeled example` = features + label (used for training)
    * Email with mark "spam/not spam"
* `Inference` - applying trained model to unlabeled examples  
* `Preprocess` - get data ready (clean, transform, reduce)
    * In image recognition - remove noise, scale down, convert to black & white
* `Loss` - how bad is model prediction on specific example?
    * e.g. 0 - prediction is perfect, 0.05 - pretty close, 0.9 - not even close
* `Loss function` - calculates loss (to adjust weights)
    * e.g. 0 - weights stays the same, 0.05 - adjusts a little, 0.9 adjusts a lot
* `Empirical risk minimization` - process of building a model with minimal loss (in supervised learning)
* `Convergence` reached if additional training on the current data will not improve the model
* `Generalization` - model's ability to adapt to new data
    * To maintain - divide data set into:
        * `Training set` (~70%) - to train models
        * `Validation set` (~15%) - compare different models, tweak parameters. Repeat multiple times, then pick the best model
        * `Test set` (~15%) - test how model reacts to new data (no more model tuning after this point)
* `Overfitting` - model predictions work on training data but fail on new data (appears when model is too complex)
    * `Regularization` - penalizing the complexity of a model to reduce overfitting
    * `Dropout` - removes random units each gradient step (ends up with a large amount of small networks)
    * `L1` - drives weights of barely relevant features to 0
    * `L2` - drives outlier weights closer to 0
    * `Regularization rate` / `lambda` - importance of the regularization function
* `Logistic regression` - generates probability (value between 0..1, e.g 0.95)
* `Activation functions`
    * `ReLU` - positive input passed on, negative converted to 0 (good default)
    * `Sigmoid function` - converts regression output to probability
* `Epoch` - training pass over entire dataset
* `Cleaning data`
    * `Clipping` - (e.g. normal range is 30-60, 12 becomes 30, 88 becomes 60)
    * `Bucketing`/`Bining` - (e.g. buckets - 0, 15, 30; 6.73 goes to 0 bucket, 25.43 goes to 30 bucket)
    * `Scrubbing` - removal of bad data (duplicates, nulls, impossible values)
    * `Scaling` - converting values into standard range (e.g. -1..1)
        * All the features should be on the same scale
* `Synthetic feature` - new feature created from existing ones
* `Model training` - `static` (trained once, used for a while), `dynamic` (trained continuously)
* `Inference` - `static` (make all possible predictions and cache them), `dynamic` (predictions on demand)
* `Preprocessing layer` - converts raw data into a format for processing (must have for model portability)
* `Data augmentation` - increase diversity of training set by applying random transformations
    * For images - rotation, rescale, brightness change, horizontal flip (ImageDataGenerator)
    * For text - [TextAttack](https://github.com/QData/TextAttack)
* `Integrated gradients` - Explainable AI technique (for debugging)
* `Logits` - vector of raw (non-normalized) predictions that a classification model generates
* `Softmax` - assigns probabilities to each class in a multi-class problem 
    * e.g. 0.95 for a dog, 0.04 for a cat, 0.01 for a bird)
* `Undersampling` - when label has a small amount of samples (unbalanced data)
    * Rare data will have almost no effect on weights, network won't learn them
    * `SMOTE` - solution to `Undersampling` (generates new examples to improve data balance)

#### Network types
* `Deep learning` - multiple layers in the network to progressively extract higher-level features (edges => nose, eyes => face)
    * `CNN` (Convolutional NN) - inspired by animal visual cortex. Each neuron in one layer connected to all neurons in the next layer
        * Usage - recognition, classification, recommendation, language processing
    * `RNN` (Recurrent  NN) - connections between nodes form a directed graph along a temporal sequence
        * Usage - modeling sequence data (time series, natural language..)
    * `DBN` (Deep Belief Network) - each layer acts as a feature detector
        * Usage - generation, recognition
* `GAN` (Generative Adversarial Network) - 
* `Autoencoder` - encodes data by ignoring noise; reconstructs original data from encoded one

#### Approaches
* `Supervised learning` - training on labeled examples (given by a "teacher"), goal is to learn rules that maps inputs to outputs
* `Unsupervised learning` - no labels, used to discover hidden patterns/feature learning
* `Reinforcement learning` - interaction with dynamic environment (e.g. move car from A to B)
    * With constant feedback (analogous to rewards) that it tries to optimize

#### Problem types
* `Analytics` - extract information from text (e.g. what is the capital of India?)
* `Regression` - predict values (e.g. what this stock price will be tomorrow?)
* `Classification` - classify objects (e.g. what this image represents? Is this email spam?)
    * Binary, multi-class
* `Recognition` - (e.g. what's written here?)
    * aka `Structured output`
* `Detection`
* `Generation` - generates new examples (e.g. create a poem please)
* `Recommenders` - create recommendations (e.g. what movies this person would like?)
    * aka `Association rule learning`
* `Clustering` - group similar examples (e.g. recognize communities within large group of people)
* `Anomaly detection` - find unusual occurrences (e.g. is something suspicious happened?)
* `Ranking` - Identify position on a scale (e.g. google search result ranking)

#### Model layers
* `Types`
    * `Input Layer` 
        * Number of neurons comprising this layer should be equal to the number of features 
    * `Hidden layers`
        * 2 layers is a good starting point
        * Number of neurons = mean of the neurons in input and output layers
        * Add `Dropout` after every layer (0.1 is good default)
    * `Output Layer` 
        * If regression or binary classification: single node. If multi-class: nodes count = labels count.
* `Layers`:
    * `Embedding` - Turns positive integers (indexes) into dense vectors of fixed size
        * e.g. `20 -> [0.6, -0.2]`
    * `GlobalAveragePooling1D` - pooling, if entire sequence is important
    * `GlobalMaxPooling1D` - pooling, if only part of sequence is important 
        * e.g. looking for a specific feature
    * `Dense` - regular densely-connected layer
    * `Dropout` - prevents overfitting 
    * `Flatten` - flattens input (e.g. 2D array to 1D)

#### Misc
* Frameworks
    * By abstraction level: `Keras` > `Tensorflow` > `Pytorch`
* Python libraries:
    * `numpy` - arrays and operations
    * `pandas` - data analysis and manipulation
    * `matplotlib` - data visualization
* `Deepfake` - media in which a person replaced with someone else
* `GPT-3` (Generative Pre-trained Transformer 3) - state-of-the-art language model (year 2020)
    * Produces human-like text (poetry, books, computer code etc)
    * [Turing test](https://lacker.io/ai/2020/07/06/giving-gpt-3-a-turing-test.html)
* [Colab](https://colab.research.google.com/) - Write and execute Python in browser (great for ML)
* [Kaggle](https://www.kaggle.com/) - Competitions, datasets, notebooks
* [Glossary](https://developers.google.com/machine-learning/glossary)
* [Playground](http://playground.tensorflow.org/) - check how network learns with different parameters
* Repository of trained ML models:
    * https://www.tensorflow.org/hub
    * https://modelzoo.co/
    * https://github.com/PaddlePaddle/PaddleHub
    

#### Best practices
* [Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml)
* [Good Data Analysis](https://developers.google.com/machine-learning/guides/good-data-analysis)
