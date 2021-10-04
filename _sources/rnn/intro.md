Lab10: RNN for NLP
=========================

This set of notebooks introduce the vanilla Elman Recurrent Neural Networks (RNNs) and how to use an Elman RNN as the featuriser for classification tasks.

We first introduce the basics of an Elman RNN in PyTorch, and illustrate how it can be used in a binary classification of simple synthetic sequences. Then we apply this a character sequence modelling for surname nationality classification with the objective of introducing general sequence modelling - where the last hidden state represent the sequence in the general sense:

- A word is a sequence of characters
- A sentence is a sequence of words
- A document is a sequence of sentences

Credit: the notebooks are adapted from:

- Chpater 8 of [Deep Learning with PyTorch Step-by-Step](https://github.com/dvgodoy/PyTorchStepByStep)
- Chapter 6 of [Natural Language Processing with PyTorch](https://github.com/joosthub/PyTorchNLPBook/tree/master/chapters/chapter_6)

