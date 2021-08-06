Lab03: spaCy NLP pipelines
===================================
spaCy makes use of two core types of objects to support various NLP tasks. 

A **container** object in spaCy groups multiple elements into a single
unit. It can be a collection of objects, like tokens or sentences, or a
set of annotations related to a single object.

**Pipeline components** objects that process the text input to
create containers and fill them with relevant data, such as a
part-of-speech tagger, a dependency parser and an entity recogniser.

In this lab, we will look at these two types of objects in spaCy to get a more in-depth understanding of how spaCy NLP code works. The we will test out our information extraction skills by deploying a simple rule-based chatbot.

**Reference:** (The book code works with spaCy v2.2, our lab code is compiled on spaCy v3.0.5)

![image](../images/spacy-book-cover.jfif)