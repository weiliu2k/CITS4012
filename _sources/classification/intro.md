Lab08: Document Classification
===============================

In this labe we will 

- first use a toy example to [revisit a simple perceptron](perceptron.ipynb) (one hidden layer neural network) in PyTorch to illustrate many fundamental concepts (activation functions, hidden layers, backpropogation, training routines) with visualisation.
- then by introducing [the PyTorch built-in `Dataset` and `Dataloader` classes](data_prep.ipynb), we take an object oriented coding approach to factorise our earlier code into a better more modulised style. This new code not only allows for the separation of data preparation from training, it also enables the transition from whole dataset training (batch gradient decent) to mini-batch gradient decent.
- lastly, we repeat the above exercises, but this time with a realworld task and* natural language dataset*: to classify whether restaurant reviews on Yelp are positive or negative using a perceptron for supervised training.

Credit: the notebooks are adapted from Chapter 3 of [Natural Language Processing with PyTorch](https://github.com/joosthub/PyTorchNLPBook/tree/master/chapters/chapter_3)

```{figure} ../images/nlp_pytorch_book.jpg
:alt: Atifical Neuron Animation
:class: bg-primary mb-1
:width: 400px
```