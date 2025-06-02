# everything about token attributes and their functions
import spacy
nlp = spacy.load("en_core_")
doc = nlp('I have two bananas and i have one $.')
for token in doc:
    print(token)
for token in doc:
    print(token, '=>', 'index: ', token.i,
          'is alpha: ', token.is_alpha,
          'is punctuated: ', token.is_punct,
          'is number: ', token.like_num,
          'is currency: ', token.is_currency)
