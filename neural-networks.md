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
    * [Turing test](https://lacker.io/ai/2020/07/06/giving-gpt-3-a-turing-test.html)

## Apps
#### Misc
* `OpenAI API`
    * `Semantic Search` - searching over documents based on the natural-language queries
    * `Customer Service` - search + chat
    * `Generation` - generate complex and consistent natural language (creative writing)
    * `Productivity Tools` - code completion, expanding content
    * `Content Comprehension` - text summary
    * `Polyglot` - text translation
    * Poor english as input, great english as output
    * Unstructured data as input, json objects as output
    * Speech to bash (e.g. "Firewall all incoming connections to port 22")
* `Experiments with Google`
    * `teachable-machine` - generate model without coding
    * 
    * 
* `Runwayml` - a lot of different ML models (SaaS)
    * `Green screen` cut any objects out of your videos
    * `Generative media` - generate video and images
    * Upscale images
* [Typical Applications](https://en.wikipedia.org/wiki/Applications_of_artificial_intelligence)
* [GPT-3 demos](https://gpt3demo.com/)

#### Text
* `TLDR This` / `Grok` - text summary
* `OthersideAI` / `Flowrite` - list of facts as input, entire article/email as output
* 

#### Images/Video
* `Dall-e` - generates images from text description
* `DeepFaceLab` - replaces faces, decreases age
* `faceswap` - replaces faces 
* `StyleGAN` - generates images (faces, cars, cats etc..)
* `Artbreeder` - combine/edit images
* `Deep Nostalgia` - Animates faces based on a photo
* `deepart.io` - repaint an image in any style
* `Wombo.ai` - lip sync based on a photo
* `Quick, Draw!` - recognizes doodle that you drew
* `Prisma` - apply filters to an image
* `Reface` - replace face in video with your face
* `Faceapp` - modify your face / change background
* `airpaper.ai` - extract data from photo of document
* `ToonMe` - face filters
* 
* `GauGAN` - you draw shapes, it generates images (landscapes)
* `Nvidia image inpainting` - choose parts of image for AI to retouch
* `Findface` - person identification based on photo 
* `PoseNet` - detects pose (used in dancing games)

#### Music generation
* `OpenAI jukebox`
* `OpenAI musenet`
* `AIVA`
* `DADABOTS`

#### Games
* `AI Dungeon` - text based rpg
* `Modbox ReplicaAI` - talk to NPCs using voice chat

#### Software development
* `losslesshq.com` - English to regex
* `GPT-3 Tailwind CSS` - English to CSS
* `bionicai.app` - English to web app component
* `Enzyme` - English to web page
* `Code Oracle` - Source code to plain english

#### Etc
* `Replika` / `Quickchat` / `bionicai.app` - chat bots
* `Siri` / `Cortana` / `Alexa` / `Google assistant` - personal assistants
* `Ask me anything` - search engine with plain text responses
