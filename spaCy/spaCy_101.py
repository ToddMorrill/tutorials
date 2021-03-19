# python3 -m spacy download en_core_web_sm
# python3 -m spacy download en_core_web_md
import spacy
from spacy.tokens import Doc
from spacy.vocab import Vocab
from spacy import displacy

nlp = spacy.load('en_core_web_md')
doc = nlp('Apple is looking at buying U.K. startup for $1 billion.')
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
          token.shape_, token.is_alpha, token.is_stop)
print()
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
print()
tokens = nlp("dog cat banana afskfsd")
for token in tokens:
    print(token.text, token.has_vector, token.vector_norm, token.is_oov)
print()
doc1 = nlp('I like salty fries and hamburgers.')
doc2 = nlp('Fast food tastes very good.')
# Similarity of two documents
print(doc1, '<->', doc2, doc1.similarity(doc2))
# Similarity of tokens and spans
french_fries = doc1[2:4]
burgers = doc1[5]
print(french_fries, '<->', burgers, french_fries.similarity(burgers))
print()
doc = nlp('I love coffee')
print(doc.vocab.strings['coffee'])  # 3197928453018144401
print(doc.vocab.strings[3197928453018144401])  # 'coffee'
print()
doc = nlp('I love coffee')
for word in doc:
    lexeme = doc.vocab[word.text]
    print(lexeme.text, lexeme.orth, lexeme.shape_, lexeme.prefix_,
          lexeme.suffix_, lexeme.is_alpha, lexeme.is_digit, lexeme.is_title,
          lexeme.lang_)
print()
doc = nlp("I love coffee")  # Original Doc
print(doc.vocab.strings["coffee"])  # 3197928453018144401
print(doc.vocab.strings[3197928453018144401])  # 'coffee' 👍

empty_doc = Doc(Vocab())  # New Doc with empty Vocab
# empty_doc.vocab.strings[3197928453018144401] will raise an error :(

empty_doc.vocab.strings.add("coffee")  # Add "coffee" and generate hash
print(empty_doc.vocab.strings[3197928453018144401])  # 'coffee' 👍

new_doc = Doc(doc.vocab)  # Create new doc with first doc's vocab
print(new_doc.vocab.strings[3197928453018144401])  # 'coffee' 👍
# displacy.serve(doc, style='ent')