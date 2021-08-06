NLP Pipelines
==================

## Traditional NLP Pipeline in NLTK

![image](../images/ie-architecture.png)

Image credit of [Information Extraction in NLTK](http://www.nltk.org/howto/relextract.html#:~:text=Relation\%20Extraction,The\%20sem.)

import nltk
nltk.download('punkt') # Sentence Tokenize
nltk.download('averaged_perceptron_tagger') # POS Tagging
nltk.download('maxent_ne_chunker') # Named Entity Chunking
nltk.download('words') # Word Tokenize

# texts is a collection of documents.
# Here is a single document with two sentences.
texts = [u"A storm hit the beach in Perth. It started to rain."]
for text in texts:
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
      words = nltk.word_tokenize(sentence)
      tagged_words = nltk.pos_tag(words)
      ne_tagged_words = nltk.ne_chunk(tagged_words)
      print(ne_tagged_words)

## Visualising NER in spaCy

We can use the `Doc.user_data` *attribute* to set a title for the visualisation.

from spacy import displacy
doc.user_data['title'] = "An example of an entity visualization"
displacy.render(doc, style='ent')

## Write the visualisation to a file

We can inform the render to not display the visualisation in the Jupyter Notebook instead write into a file by calling the render with two extra argument:

```
jupyter=False, page=True
```

from pathlib import Path
# the page=True indicates that we want to write to a file
html = displacy.render(doc, style='ent', jupyter=False, page=True)
output_path = Path("C:\\Users\\wei\\CITS4012\\ent_visual.html")
output_path.open("w", encoding="utf-8").write(html)

## NLP pipeline in spaCy

Recall that spaCy's container objects represent linguistic units, suchas a text (i.e. document), a sentence and an individual token withlinguistic features already extracted for them.

How does spaCy create these containers and fill them withrelevant data?

A spaCy pipeline include, by default, a part-of-speech tagger (`tagger`), a dependency parser (`parser`), a lemmatizer (`lemmatizer`), an entity recognizer (`ner`), an attribute ruler (`attribute_ruler` and a word vectorisation model (`tok2vec`)).

import spacy
nlp = spacy.load('en_core_web_sm')
nlp.pipe_names

spaCy allows you to load a selected set of pipeline components, dis-abling those that aren't necessary.

You can do this either when creating a nlp object or disable it after the nlp object is created. 

```{tabbed} Disabling when create
`nlp = spacy.load('en_core_web_sm',disable=['parser'])`
```

```{tabbed} Disabling after creation
`nlp.disable_pipes('tagger')`

`nlp.disable_pipes('parser')`
```

## Customising a NLP pipe in spaCy

import spacy
nlp = spacy.load('en_core_web_sm')

doc = nlp(u'I need a taxi to Cottesloe.')
for ent in doc.ents:
    print(ent.text, ent.label_)

```{admonition} What if?
If we would like to introduce a new entity type `SUBURB` for `Cottesloe` and other suburb names, how should we informthe NER component about it?
```

*Steps of Customising a spaCy NER pipe*
1. Create a training example to show the entity recognizer so it will learn what to apply the SUBURB label to;
2. Add a new label called SUBURB to the list of supported entitytypes;
3. Disable other pipe to ensure that only the entity recogniser will beupdated during training;
4. Start training;
5. Test your new NER pipe;
6. Serialise the pipe to disk;
7. Load the customised NER

import spacy
nlp = spacy.load('en_core_web_sm')

# Specify new label and training data
LABEL = 'SUBURB' 
TRAIN_DATA = [('I need a taxi to Cottesloe', 
                { 'entities': [(17, 26, 'SUBURB')] }),
              ('I like red oranges', { 'entities': []})]

# Add new label to the ner pipe
ner = nlp.get_pipe('ner')
ner.add_label(LABEL)

# Train
optimizer = nlp.create_optimizer() 
import random
from spacy.tokens import Doc
from spacy.training import Example
for i in range(25):
  random.shuffle(TRAIN_DATA)
  for text, annotations in TRAIN_DATA:
    doc = Doc(nlp.vocab, words=text.split(" "))
    # We need to create a training example object
    example = Example.from_dict(doc, annotations)
    nlp.update([example], sgd=optimizer)

# Test
doc = nlp(u'I need a taxi to Crawley')
for ent in doc.ents:
  print(ent.text, ent.label_)

# Serialize the entire model to disk
nlp.to_disk('C:\\Users\\wei\\CITS4012') # Windows Path

# Load spacy model from disk
import spacy
nlp_updated = spacy.load('C:\\Users\\wei\\CITS4012')

# Test
doc = nlp_updated(u'I need a taxi to Subiaco')
for ent in doc.ents:
  print(ent.text, ent.label_)

:::{admonition} Your Turn 
- Replace the suburb name with a few others, for example 'Claremont', 'Western Australia' and see what the the entity label is. 
- Take a look at the directory and see how the `nlp` model is stored. 
- This [blog post on How to Train spaCy to Autodetect New Entities (NER) [Complete Guide]](https://www.machinelearningplus.com/nlp/training-custom-ner-model-in-spacy/) has more extensive examples on how to train a ner model with more data. 
:::