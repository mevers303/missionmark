# Mark Evers
# 5/16/18
# nlp.py
# Script for model extraction


import os
import sys
sys.path.append(os.getcwd())
sys.path.append("src")


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from src.data import get_cached_corpus_filenames
from src.globals import *

from src.pickle_workaround import pickle_dump, pickle_load

import re
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

    tokens = [stemmer.stem(token) for token in re.split(r"[^a-zA-Z]+", text) if token]

    docs_completed += 1
    progress_bar(docs_completed, n_docs)

    return tokens


def count_vectorize(doc_ids, corpus, table_name, input="content"):

    count_vectorizer_corpus = None

    if MODEL_PICKLING and os.path.exists(f"data/{table_name}/pickles/CountVectorizer.pkl"):
        debug("Loading cached vectorizer...")
        count_vectorizer = pickle_load(f"data/{table_name}/pickles/CountVectorizer.pkl")

    else:
        debug("Vectorizing documents...")
        count_vectorizer = CountVectorizer(input=input, stop_words=get_stopwords(), tokenizer=tokenize, ngram_range=(1,1), strip_accents="ascii", dtype=np.uint16)
        count_vectorizer_corpus = count_vectorizer.fit_transform(corpus)
        debug(" -> Done!", 1)

        debug("Caching vectorizer...")
        pickle_dump(count_vectorizer, f"data/{table_name}/pickles/CountVectorizer.pkl")
        debug(" -> Vectorizer cached!", 1)

    debug(f" -> Loaded vectorizer with {len(count_vectorizer.get_feature_names())} features!", 1)


    if not count_vectorizer_corpus and CORPUS_PICKLING and os.path.exists(f"data/{table_name}/pickles/CountVectorizer_corpus.pkl"):
        debug("Loading cached count vector...")
        count_vectorizer_corpus = pickle_load(f"data/{table_name}/pickles/CountVectorizer_corpus.pkl")
        with open(f"data/{table_name}/pickles/CountVectorizer_doc_ids.txt", "r") as f:
            doc_ids = [line[:-1] for line in f]
    else:
        debug("Transforming corpus...")
        count_vectorizer_corpus = count_vectorizer.transform(corpus)
        debug(" -> Done!", 1)

        debug("Caching corpus count vector...")
        pickle_dump(count_vectorizer_corpus, f"data/{table_name}/pickles/CountVectorizer_corpus.pkl")
        with open(f"data/{table_name}/pickles/CountVectorizer_doc_ids.txt", "w") as f:
            for doc_id in doc_ids:
                f.write(doc_id + "\n")
        debug(" -> Corpus vector cached!", 1)

    debug(f" -> {count_vectorizer_corpus.shape[0]} tokens found!", 1)


    return count_vectorizer, doc_ids, count_vectorizer_corpus


def count_vectorize_cache():

    global n_docs

    corpus_filenames, n_docs = get_cached_corpus_filenames("govwin_opportunity")
    doc_ids = [file[:-4] for file in corpus_filenames]

    count_vectorizer, doc_ids, count_vectorizer_corpus = count_vectorize(doc_ids, corpus_filenames, "fbo_files", input="filename")

    return count_vectorizer, count_vectorizer_corpus


if __name__ == "__main__":
    count_vectorize_cache()
