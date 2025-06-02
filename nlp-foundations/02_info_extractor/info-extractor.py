import spacy
nlp = spacy.load('en_core_web_sm')
text = '''Invoice Total: $1,200.00  
Please contact finance@example.com for queries.  
More details at: https://company.com/invoice/123
'''
doc = nlp(text)
emails = []
urls = []
numbers = []
for token in doc:
    if token.like_email:
        emails.append(token.text)
    elif token.like_url:
        urls.append(token.text)
    elif token.like_num:
        numbers.append(token.text)


print("Emails:", emails)
print("URLs:", urls)
print("Numbers:", numbers)
