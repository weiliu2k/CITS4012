Exercise
=========

**Extend a `to + GPE` NER pattern**

Let's consider a travel assistant here. Write a function that uses the entity type `GPE` to find the desired destination of a user. The code below is capable of very simple parsing, but not able to handle sentences like `I am going to a conference in Berlin.` 

- Modify the code to make it work for more cases. 
- Incoporate that into your Telgram booking bot. 

import spacy
nlp = spacy.load('en_core_web_sm')

# Here's the function that figures out the destination
def det_destination(doc):
    for i, token in enumerate(doc):
        if token.ent_type != 0 and token.ent_type_ == 'GPE':
            while True:
                token = token.head
                if token.text == 'to':
                    return doc[i].text
                if token.head == token:
                    return 'Failed to determine'
    return 'Failed to determine'

# Testing the det_destination function
doc = nlp(u'I am going to Berlin.')
# doc = nlp(u'I am going to the conference in Berlin.')
for token in doc:
    print(token.text, token.ent_type_, token.head)

dest = det_destination(doc)
print('It seems the user wants a ticket to ' + dest)