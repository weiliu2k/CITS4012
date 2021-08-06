# Load spacy without NER
import spacy
from spacy.pipeline import EntityRecognizer
nlp = spacy.load('en_core_web_sm', disable=['ner'])
nlp.pipe_names

TOKEN = '1947674606:AAHIc6MAzOWAOivUFXjkrsV6MsFWQcw13oY'

#ner = nlp.create_pipe('ner')
ner = EntityRecognizer(nlp.vocab, )
ner.from_disk('C:\\Users\\wei\\CITS4012')
nlp.add_pipe(ner)
nlp.pipe_names

