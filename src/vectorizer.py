# Mark Evers
# 5/16/18
# nlp.py
# Script for model extraction


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from src.data import *
from src.globals import *

import pickle

import re
import os
from nltk.stem.porter import PorterStemmer as Stemmer

import numpy as np


n_docs = 0
docs_completed = 0



def get_stopwords():
    """
    Reads the stopwords from stopwords.txt and combines them with the ENGLISH_STOP_WORDS in sklearn.
    :return: A set of stopwords.
    """

    with open("stopwords.txt", "r") as f:
        custom_stopwords = {word[:-1] for word in f}

    return list(ENGLISH_STOP_WORDS.union(custom_stopwords))


stemmer = Stemmer()
def tokenize(text):
    """
    Tokenizes a document to be converted to a TF-IDF vector.
    :param text: The text/document to be tokenized.
    :return: A list of stemmed tokens.
    """

    global n_docs, docs_completed

    tokens = [stemmer.stem(token) for token in split_tokens_hard(text)]

    docs_completed += 1
    progress_bar(docs_completed, n_docs)

    return tokens


def split_tokens_hard(text):
    """
    Splits into tokens based on any character that is NOT a letter, number, "-", or ".".
    :param text: The text to be tokenized
    :return: A list of tokens
    """
    # return [token for token in re.split(r"[^a-zA-Z0-9\-\.]+|[\-\.]{3,}|\s[\-\.]+|[\-\.]+\s", text) if token]  # list comprehension removes empty strings
    # return [token for token in re.split(r"[^a-zA-Z0-9]+", text) if token]  # list comprehension removes empty strings
    return [token for token in re.split(r"[^a-zA-Z]+", text) if token]  # list comprehension removes empty strings


def count_vectorize(corpus, input="content"):

    if MODEL_PICKLING and os.path.exists("data/pickles/CountVectorizer.pkl") and os.path.exists("data/pickles/CountVectorizer_corpus.pkl"):
        debug("Loading cached vectorizer...")

        with open("data/pickles/CountVectorizer.pkl", "rb") as f:
            count_vectorizer = pickle.load(f)

        with open("data/pickles/CountVectorizer_corpus.pkl", "rb") as f:
            count_vectorizer_corpus = pickle.load(f)

    else:
        debug("Vectorizing documents...")

        count_vectorizer = TfidfVectorizer(input=input, stop_words=get_stopwords(), tokenizer=tokenize, max_df=.66, min_df=2, ngram_range=(1,1), sublinear_tf=True, strip_accents="ascii", dtype=np.uint16)
        count_vectorizer_corpus = count_vectorizer.fit_transform(corpus)

        debug("Caching vectorizer...")
        with open("data/pickles/CountVectorizer.pkl", "wb") as f:
            pickle.dump(count_vectorizer, f)
        with open("data/pickles/CountVectorizer_corpus.pkl", "wb") as f:
            pickle.dump(count_vectorizer_corpus, f)
        debug(" -> Vectorizer cached!", 2)

    debug(f" -> {count_vectorizer.shape[1]} tokens found!", 1)
    return count_vectorizer, count_vectorizer_corpus


def count_vectorize_cache():

    global n_docs

    debug("Searching for cached documents...")
    corpus_filenames = get_cached_corpus_filenames()
    n_docs = len(corpus_filenames)
    debug(f" -> {n_docs} cached documents found!", 2)

    debug("Vectorizing documents...")
    count_vectorizer, count_vectorizer_corpus = count_vectorize(corpus_filenames, input="filename")
    debug(f" -> {n_docs} documents vectorized!", 2)

    return count_vectorizer, count_vectorizer_corpus


if __name__ == "__main__":
    count_vectorize_cache()
