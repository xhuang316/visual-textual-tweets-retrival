# Visual-textual-tweets-retrival
This project presents an automatic approach to labelling on-topic social media posts using a visual-textual fused vector. Two convolutional neural networks (CNNs), the Inception-V3 CNN and word embedded CNN, are applied to extract visual and textual features, respectively, from social media posts. Well-trained on our designed training sets, the extracted visual and textual features are further concatenated to form a fused feature to feed the final classification process.

## Getting Started

This implementation is based on Python coding environment. A high-level neural network API, Keras (https://keras.io/), is used to enable fast experimentation. The code requires python 3.6 compiler and the installation of Keras with TensorFlow backend. The authors recommend the usage of GPU-supported TensorFlow to reduce the training time. 

## Prerequisites

Python 3.6.5

GPU supported TensorFlow

CUDA v9.0

## Training word2vec
Given the fact that textual patterns differ a lot in short-text posts in social media compared to formal sources including news and formal articles, it is necessary to train word vectors specifically for social media posts. To acquire word vectors, the technique used in this study is Word2Vec, a shallow neural network with single hidden layer, but proved to be powerful in providing 300-dimention vectors representing the word characteristics. Use Word2vec_training or Word2vec_training_separated_CSV to train the network in order to assign vectors to different words. Word2vec_training accepts single CSV as corpus input while Word2vec_training_separated_CSV accepts separated CSV as corpus input. 

## Training word embedded CNN

## Transfer learning Inception V3

## Fused CNN architecture

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* **Xiao Huang** - Department of Geography, University of South Carolina

