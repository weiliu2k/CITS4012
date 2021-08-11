Document Classification
=======================
Now that we have a good understanding of TF-IDF term document matrix, we can treat each term as a feature, and each document (row) as an instance or a training sample to train a classifier. The classifier can be any traditional supervised learning models that deals with tabular shaped data, where one colum stores the labels of each sample record. 

The goal of this guide is to explore some of the main 'scikit-learn'
tools on a popular classification task: analyzing a collection of text
documents (newsgroups posts) and classify them into one of the twenty different topics.

In this notebook we will see how to:
 - load the file contents and the categories
 - extract feature vectors suitable for machine learning
 - train a linear model to perform categorization
 - use a grid search strategy to find a good configuration of both the feature extraction components and the classifier

Original Notebook Credit to [scikit-learn tutorial on Working with Text Data](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)

## Loading the 20 newsgroups dataset

The dataset is called "Twenty Newsgroups". Here is the [official
description](http://people.csail.mit.edu/jrennie/20Newsgroups/):

> The 20 Newsgroups data set is a collection of approximately 20,000
  newsgroup documents, partitioned (nearly) evenly across 20 different
  newsgroups. To the best of our knowledge, it was originally collected
  by Ken Lang, probably for his paper "Newsweeder: Learning to filter
  netnews," though he does not explicitly mention this collection.
  The 20 newsgroups collection has become a popular data set for
  experiments in text applications of machine learning techniques,
  such as text classification and text clustering.

In the following we will use the built-in dataset loader for 20 newsgroups
from scikit-learn. 

In order to get faster execution times for this first example we will
work on a partial dataset with only 4 categories out of the 20 available
in the dataset:

categories = ['alt.atheism', 'soc.religion.christian',
               'comp.graphics', 'sci.med']

We can now load the list of files matching those categories as follows (this may take a while - 65.1s on a desktop computer - AMD 16 core, 32GB RAM):

from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train',
      categories=categories, shuffle=True, random_state=42)


The returned dataset is a `scikit-learn` "bunch": a simple holder
object with fields that can be both accessed as python `dict`
keys or `object` attributes for convenience, for instance the
`target_names` holds the list of the requested category names::

twenty_train.target_names

The files themselves are loaded in memory in the `data` attribute. For
reference the filenames are also available:

len(twenty_train.data)

twenty_train.filenames[0]

Let's print the first three lines of the first loaded file:

print("\n".join(twenty_train.data[0].split("\n")[:3]))

Below is how to access the class label (i.e. target column) of the first document. 

print(twenty_train.target_names[twenty_train.target[0]])

Supervised learning algorithms will require a category label for each
document in the training set. In this case the category is the name of the
newsgroup which also happens to be the name of the folder holding the
individual documents.

For speed and space efficiency reasons `scikit-learn` loads the
target attribute as an array of integers that corresponds to the
index of the category name in the `target_names` list. The category
integer id of each sample is stored in the `target` attribute:

twenty_train.target[:10]

It is possible to get back the category names as follows:

for t in twenty_train.target[:10]:
    print(twenty_train.target_names[t])

You might have noticed that the samples were shuffled randomly when we called
`fetch_20newsgroups(..., shuffle=True, random_state=42)`: this is useful if
you wish to select only a subset of samples to quickly train a model and get a
first idea of the results before re-training on the complete dataset later.

## Extracting features from text files

In order to perform machine learning on text documents, we first need to
turn the text content into numerical feature vectors.

### Bags of words

The most intuitive way to do so is to use a bags of words representation:

  1. Assign a fixed integer id to each word occurring in any document
     of the training set (for instance by building a dictionary
     from words to integer indices).

  2. For each document `#i`, count the number of occurrences of each
     word `w` and store it in `X[i, j]` as the value of feature
     `#j` where `j` is the index of word `w` in the dictionary.

The bags of words representation implies that `n_features` is
the number of distinct words in the corpus: this number is typically
larger than 100,000.

If `n_samples == 10000`, storing `X` as a NumPy array of type
float32 would require 10000 x 100000 x 4 bytes = **4GB in RAM** which
is barely manageable on today's computers.

Fortunately, **most values in X will be zeros** since for a given
document less than a few thousand distinct words will be
used. For this reason we say that bags of words are typically
**high-dimensional sparse datasets**. We can save a lot of memory by
only storing the non-zero parts of the feature vectors in memory.

`scipy.sparse` matrices are data structures that do exactly this,
and `scikit-learn` has built-in support for these structures.

### Tokenizing text with `scikit-learn`

Text preprocessing, tokenizing and filtering of stopwords are all included
in class: `CountVectorizer`, which builds a dictionary of features and
transforms documents to feature vectors:

 from sklearn.feature_extraction.text import CountVectorizer
 
 count_vect = CountVectorizer()
 X_train_counts = count_vect.fit_transform(twenty_train.data)
 X_train_counts.shape


Class `CountVectorizer` supports counts of N-grams of words or consecutive
characters. Once fitted, the vectorizer has built a dictionary of feature
indices:

# The number of times the word 'algorithm' occurs 
count_vect.vocabulary_.get(u'algorithm')

The index value of a word in the vocabulary is linked to its frequency
in the whole training corpus.

:::{note}

  The method `count_vect.fit_transform` performs two actions:
  it learns the vocabulary and transforms the documents into count vectors.
  It's possible to separate these steps by calling
  `count_vect.fit(twenty_train.data)` followed by
  `X_train_counts = count_vect.transform(twenty_train.data)`,
  but doing so would tokenize and vectorize each text file twice.
:::

### From occurrences to frequencies

Occurrence count is a good start but there is an issue: longer
documents will have higher average count values than shorter documents,
even though they might talk about the same topics.

To avoid these potential discrepancies it suffices to divide the
number of occurrences of each word in a document by the total number
of words in the document: these new features are called `tf` for Term
Frequencies.

Another refinement on top of tf is to downscale weights for words
that occur in many documents in the corpus and are therefore less
informative than those that occur only in a smaller portion of the
corpus.

This downscaling is the `tf–idf` -  "Term Frequency times
Inverse Document Frequency" we discussed earlier. 

As dicussed in the previous notebooks, both **tf** and **tf–idf** can be computed as follows using class `TfidfTransformer`:

from sklearn.feature_extraction.text import TfidfTransformer

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape

In the above example-code, we firstly use the `fit(..)` method to fit our
estimator to the data and secondly the `transform(..)` method to transform
our count-matrix to a tf-idf representation.
These two steps can be combined to achieve the same end result faster
by skipping redundant processing. This is done through using the
`fit_transform(..)` method as shown below, and as mentioned in the note
in the previous section:

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

X_train_tfidf.toarray()

## Training a classifier

Now that we have our features, we can train a classifier to try to predict
the category of a post. Let's start with a `naïve Bayes <naive_bayes>`
classifier, which provides a nice baseline for this task. `scikit-learn` includes several
variants of this classifier; the one most suitable for word counts is the
multinomial variant:

from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

To try to predict the outcome on a new document we need to extract
the features using almost the same feature extracting chain as before.
The difference is that we call `transform` instead of `fit_transform`
on the transformers, since they have already been fit to the training set:

docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
  print('%r => %s' % (doc, twenty_train.target_names[category]))

## Building a pipeline

In order to make the vectorizer => transformer => classifier easier
to work with, `scikit-learn` provides a class `~sklearn.pipeline.Pipeline` that behaves
like a compound classifier:

from sklearn.pipeline import Pipeline
text_clf = Pipeline([
     ('vect', CountVectorizer()),
     ('tfidf', TfidfTransformer()),
     ('clf', MultinomialNB())])

The names `vect`, `tfidf` and `clf` (classifier) are arbitrary.
We will use them to perform grid search for suitable hyperparameters below.
We can now train the model with a single command:

text_clf.fit(twenty_train.data, twenty_train.target)

## Evaluation of the performance on the test set

Evaluating the predictive accuracy of the model is simply a comparision of the predicted and the actual labels. 

import numpy as np
twenty_test = fetch_20newsgroups(subset='test', 
              categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
np.mean(predicted == twenty_test.target)

We achieved 83.5% accuracy. Let's see if we can do better with a
linear `support vector machine (SVM) <svm>`,
which is widely regarded as one of
the best text classification algorithms (although it's also a bit slower
than naïve Bayes). We can change the learner by simply plugging a different
classifier object into our pipeline:

from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([
     ('vect', CountVectorizer()),
     ('tfidf', TfidfTransformer()),
     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                           alpha=1e-3, random_state=42,
                           max_iter=5, tol=None))])

text_clf.fit(twenty_train.data, twenty_train.target)

predicted = text_clf.predict(docs_test)
np.mean(predicted == twenty_test.target)

We achieved 91.3% accuracy using the SVM. `scikit-learn` provides further
utilities for more detailed performance analysis of the results:

from sklearn import metrics
print(metrics.classification_report(twenty_test.target, predicted,
      target_names=twenty_test.target_names))

metrics.confusion_matrix(twenty_test.target, predicted)

As expected the confusion matrix shows that posts from the newsgroups
on atheism and Christianity are more often confused for one another than
with computer graphics.

:::{note}

SGD stands for Stochastic Gradient Descent. This is a simple
optimization algorithms that is known to be scalable when the dataset
has many samples.

By setting `loss="hinge"` and `penalty="l2"` we are configuring
the classifier model to tune its parameters for the linear Support
Vector Machine cost function.
:::

Alternatively we could have used `sklearn.svm.LinearSVC` (Linear
Support Vector Machine Classifier) that provides an alternative
optimizer for the same cost function based on the liblinear_ C++
library.

## Parameter tuning using grid search

We've already encountered some parameters such as `use_idf` in the
`TfidfTransformer`. Classifiers tend to have many parameters as well;
e.g., `MultinomialNB` includes a smoothing parameter `alpha` and
`SGDClassifier` has a penalty parameter `alpha` and configurable loss
and penalty terms in the objective function (see the module documentation,
or use the Python `help` function to get a description of these).

Instead of tweaking the parameters of the various components of the
chain, it is possible to run an exhaustive search of the best
parameters on a grid of possible values. We try out all classifiers
on either words or bigrams, with or without idf, and with a penalty
parameter of either 0.01 or 0.001 for the linear SVM:

from sklearn.model_selection import GridSearchCV
parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1e-2, 1e-3),
}

Obviously, such an exhaustive search can be expensive. If we have multiple
CPU cores at our disposal, we can tell the grid searcher to try these eight
parameter combinations in parallel with the `n_jobs` parameter. If we give
this parameter a value of `-1`, grid search will detect how many cores
are installed and use them all:

gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)

The grid search instance behaves like a normal ``scikit-learn``
model. Let's perform the search on a smaller subset of the training data
to speed up the computation:

gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])

After calling `fit` on a `GridSearchCV` object, we now obtained a classifier
that we can use to `predict`:

twenty_train.target_names[gs_clf.predict(['God is love'])[0]]

The object's `best_score_` and `best_params_` attributes store the best
mean score and the parameters setting corresponding to that score:

gs_clf.best_score_

for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

A more detailed summary of the search is available at `gs_clf.cv_results_`.

The `cv_results_` parameter can be easily imported into pandas as a
`DataFrame` for further inspection.

:::{note}
  A `GridSearchCV` object also stores the best classifier that it trained
  as its `best_estimator_` attribute. In this case, that isn't much use as
  we trained on a small, 400-document subset of our full training set.
:::

The index value of a word in the vocabulary is linked to its frequency
in the whole training corpus.

:::{admonition} Your Turn
Update the code to use TfidfVectorisor. Think about how a classification model or a similarity/distance calculation using TF-IDF might help your chatbot. 
:::