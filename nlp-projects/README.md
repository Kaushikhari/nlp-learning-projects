# 📱  Spam Classifier

This project builds a spam classifier using Natural Language Processing (NLP) and machine learning with Python. It classifies **SMS (text) messages** as either **spam** or **ham** (not spam) using SpaCy and scikit-learn.

---

## 🚀 Project Overview

The model analyzes short SMS messages to detect unwanted or spammy content. It uses:
- Tokenization & Lemmatization (via `spaCy`)
- Vectorization (`CountVectorizer`)
- Naive Bayes classification (`MultinomialNB`)

The dataset used is the [`spam.csv`](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset), which contains over 5,000 labeled text messages.

> _Note: This project was developed for learning purposes with guidance from online resources and communities. It combines inspiration from CodeBasics tutorials and feedback from ChatGPT._

---

## 🛠️ Tech Stack

- Python
- Pandas
- spaCy (`en_core_web_sm`)
- scikit-learn
- Matplotlib & Seaborn (for visualization)

---

## 🧹 Preprocessing

Each SMS is:
- Lowercased
- Tokenized
- Cleaned (stopwords and punctuation removed)
- Lemmatized (converted to root form)

```python
def preprocess(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
