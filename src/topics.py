# Mark Evers
# 5/7/18
# topics.py
# Script for model extraction


from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from src.data import get_test_data

import re
from nltk.stem.porter import PorterStemmer

import numpy as np


stemmer = PorterStemmer()
def stem_tokens(tokens):
    """Port-stems a list of tokens."""

    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))

    return stemmed


def tokenize(text):
    """Tokenizes a document."""

    tokens = re.split(r"[^a-zA-Z0-9\-]+|\b[0-9]+\b", text)
    stems = stem_tokens(tokens)

    return stems


def print_top_words(H, words):

    indices = np.array(np.argsort(H, axis=1)[:, :-21:-1])
    i = 0
    for topic in indices:
        print("Top words for latent feature {0}:\n".format(i), [words[word] for word in topic])
        print() # newline
        i += 1



def print_top_docs(W, titles):

    indexes = np.argsort(W, axis=0)[:-11:-1, :]

    for column in range(indexes.shape[1]):
        print("Top articles for topic {0}:".format(column))
        for i in indexes[:, column]:
            print(titles[i])
        print()


stop_words = ENGLISH_STOP_WORDS.copy().union({"-", "pdf", "docx", ""})
vectorizer = TfidfVectorizer(stop_words=stop_words, tokenizer=tokenize)
corpus = get_test_data()
corpus_tfidf = vectorizer.fit_transform(corpus)

model = NMF(n_components=25, max_iter=100)

W = model.fit_transform(corpus_tfidf)
H = model.components_
