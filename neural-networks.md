## Concepts
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
* `Deepfake` - media in which a person replaced with someone else
* `GPT-3` (Generative Pre-trained Transformer 3) - state-of-the-art language model (year 2020)
    * Produces human-like text (poetry, books, computer code etc)

## Apps
#### Misc
* `OpenAI API`
    * `Semantic Search` - searching over documents based on the natural-language queries
    * `Customer Service` - search + chat
    * `Generation` - generate complex and consistent natural language (creative writing)
    * `Productivity Tools` - code completion, expanding content
    * `Content Comprehension` - text summary
    * `Polyglot` - text translation
* `Experiments with Google`
    * `teachable-machine` - generate model without coding
    * 
    * 
* `Runwayml` - a lot of different ML models (SaaS)
* [Typical Applications](https://en.wikipedia.org/wiki/Applications_of_artificial_intelligence)

#### Text
* `TLDR This` - text summary

#### Images/Video
* `Dall-e` - generates images from text description
* `DeepFaceLab` - replaces faces, decreases age
* `faceswap` - replaces faces 
* `StyleGAN` - generates images (faces, cars, cats etc..)
* `Artbreeder` - combine/edit images
* `deepart.io` - repaint an image in any style
* `Quick, Draw!` - recognizes doodle that you drew
* `Prisma` - apply filters to an image
* `Reface` - replace face in video with your face
* `Faceapp` - modify your face / change background
* 
* `GauGAN` - you draw shapes, it generates images (landscapes)
* `Nvidia image inpainting` - choose parts of image for AI to retouch
* `Findface` - person identification based on photo 
* `PoseNet` - detects pose (used in dancing games)

#### Music
* `OpenAI jukebox` - generates music
* `OpenAI musenet` - generates music

#### Games
* `AI Dungeon` - text based rpg

#### Etc
* `Replika` - chat bot
* `Siri`, `Cortana`, `Alexa`, `Google assistant` - personal assistants

