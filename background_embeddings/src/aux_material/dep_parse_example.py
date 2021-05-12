import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("My dog is little.")
options = {"compact": True}
displacy.serve(doc, style="dep", options=options)