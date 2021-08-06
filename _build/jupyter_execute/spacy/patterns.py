Finding Patterns
================ 

Imagine you are building a chat bot and we are trying to find utterances in user input that express one of the following: 

```{tabbed} What's Expressed
ability, possibility, permission, or obligation (as opposed to utterances that describe real actions that have occurred, are occurring, or occur regularly)
```
```{tabbed} Example Sentences
For instance, we want to find “I can do it.”
but not “I’ve done it.”
```
```{tabbed} Linguistic Pattern
`subject + auxiliary + verb + . . . + direct object + ...`

The ellipses indicate that the direct object isn't necessarily
located immediately behind the verb, there might be other words in between.
```

## Check spaCy version

!pip show spacy

## Hard-coded pattern discovery

To look for the `subject + auxiliary + verb + . . . + direct object + ...` pattern programmably, we need to go through each token's dependency label (*not part of speech label*) to first find the sequence of `nsubj aux ROOT` where `ROOT` indicate the root verb, then for each of children of the root verb (`ROOT`) we check to see if it is a direct object (`dobj`) of the verb. 

import spacy
nlp = spacy.load('en_core_web_sm')
def dep_pattern(doc):
    for i in range(len(doc)-1):
        if doc[i].dep_ == 'nsubj' and doc[i+1].dep_ == 'aux' and doc[i+2].dep_ == 'ROOT':
            for tok in doc[i+2].children:
                if tok.dep_ == 'dobj':
                    return True
    
    return False

# doc = nlp(u'We can overtake them.')
doc = nlp(u'I might send them a card as a reminder.')

### Use displaycy to visualise the dependency 

from spacy import displacy
displacy.render(doc, style='dep')



options = {'compact': True, 'font': 'Tahoma'}
displacy.render(doc, style='dep', options=options)


if dep_pattern(doc):
    print('Found')
else:
    print('Not found')

:::{admonition} Code Explanation
:class: tip, dropdown
The `dep_pattern` function above takes a Doc object as parameter and returns a binary value `True` if the hard-coded pattern `subject + auxiliary + verb + . . . + direct object + ...` is found, otherwise `False`. The function iterates over the Doc object's tokens,
searching for a `subject + auxiliary + verb`, where the `verb` is the `root` of the dependency tree. If the pattern is found, then we check whether the verb has a direct object among its syntactic children. Finally, if we find a direct object, the function returns `True`.
Otherwise, it returns `False`.
:::

## Using spaCy pattern matcher

spaCy has a predefined tool called `Matcher`, that
is specially designed to find sequences of tokens based on pattern rules. An implementation of the “subject + auxiliary + verb” pattern with
`Matcher` might look like this:

import spacy
from spacy.matcher import Matcher
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)
pattern = [{"DEP": "nsubj"}, {"DEP": "aux"}, {"DEP": "ROOT"}]
matcher.add("NsubjAuxRoot", [pattern])
doc = nlp("We can overtake them.")
matches = matcher(doc)
for match_id, start, end in matches:
    span = doc[start:end]
    print("Span: ", span.text)
    print("The positions in the doc are: ", start, "-", end)
    print("Match ID ", match_id)
    print(doc.vocab.strings[match_id]) 
    for tok in doc[end-1].children:
        if tok.dep_ == 'dobj':
            print("The direct object of {} is {}".format(doc[end-1], tok.dep_))   

:::{admonition} Code Explanation
:class: tip, dropdown
spaCy `Matcher` class takes a model's vocabulary as input and creates a matcher object named `matcher`. Then we need to define a pattern of interest. The pattern is specified in a dictionary object, and the order of the key value pairs indicate the desired sequence we are trying to find a match for. Once the pattern is found, a list of tuples in the form of `(match_id, start, end)` is returned. The `match_id` is the hash value of the string ID "NsubjAuxRoot". To get the string value, you can look up the ID in the StringStore.
:::

## Summary of Rule-based Matching 

:::{admonition} Steps for using the <span>Matcher</span> class:
:class tip
1.  Create a <span>Matcher</span> instance by passing in a shared Vocab
    object;

2.  Specify the pattern as an list of dependency labels;

3.  Add the pattern to the a <span>Matcher</span> object;

4.  Input a <span>Doc</span> object to the matcher;

5.  Go through each match
    $\langle match\_id, start, end \rangle$.
:::

We have seen a *Dependency Matcher* just now, there are more Rule-based
matching support in spaCy:

-   Token Matcher: <span>regex</span>, and patterns such as

-   Phrase Matcher: <span>PhraseMatcher</span> class

-   Entity Ruler

-   Combining models with rules

For more information of different types of matchers, see [spaCy Documentation on Rule Based Matching](https://spacy.io/usage/rule-based-matching#matcher).

**Reference**: Chapter 6 of NATURAL LANGUAGE PROCESSING WITH PYTHON AND SPACY