Lab07: Word Embeddings
======================

Word Vectors are often used as a fundamental component for downstream NLP tasks, e.g. question answering, text generation, translation, etc., so it is important to build some intuitions as to their strengths and weaknesses. Here, you will explore two types of word vectors: those derived from *co-occurrence matrices*, and those derived via *GloVe*. 

:::{note}
The terms "word vectors" and "word embeddings" are often used interchangeably. The term "embedding" refers to the fact that we are encoding aspects of a word's meaning in a lower dimensional space. As [Wikipedia](https://en.wikipedia.org/wiki/Word_embedding) states, "*conceptually it involves a mathematical embedding from a space with one dimension per word (one-hot encoding) to a continuous vector space with a much lower dimension*".
:::

In this lab, we will look at how to use SVD on Word-Word co-ocurrence matrix to obtain word embeddings, and then visualising pretrained word embeddings using Matplotlib and TSNE, finally we train our own embeddings using a document crawled from the Web, and copmare with the pre-trained embeddings. 

Credit: the notebooks are adapted from [the Stanford CS224N Assignment on Exploring Word Vectors](http://web.stanford.edu/class/cs224n/assignments/a1_preview/exploring_word_vectors.html). 

## Environment Update with more packages

### Install gensim

Note this will install the latest gensim 4.x

```
conda install -c conda-forge gensim
```
### Install import-ipynb

This is a package to import functions from other Jupyter Notebooks.
```
pip install import-ipynb
```

### Install Python Levenshtein Similarity

```
pip install python-Levenshtein
```

### Install interactive visualisation bokeh

```
pip install bokeh
```

### Install bs4 to get BeautifulSoup for web crawling

```
pip install bs4
```