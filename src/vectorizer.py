# Mark Evers
# 5/16/18
# nlp.py
# Script for model extraction


import os
import sys
sys.path.append(os.getcwd())
sys.path.append("src")


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from src.progress_bar_vetorizers import CountVectorizerProgressBar
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from src.data import get_cached_filenames
from src.globals import *

from src.pickle_workaround import pickle_dump, pickle_load

import re
from nltk.stem.porter import PorterStemmer as Stemmer

import numpy as np




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
    return [stemmer.stem(token) for token in re.split(r"[^a-zA-Z]+", text) if token]


def count_vectorize(doc_ids, corpus, table_name, input_type="content"):

    count_vectorizer_corpus = None

    if MODEL_PICKLING and os.path.exists(f"data/{table_name}/pickles/CountVectorizer.pkl"):
        count_vectorizer = pickle_load(f"data/{table_name}/pickles/CountVectorizer.pkl")

    else:
        debug("Vectorizing documents...")
        count_vectorizer = CountVectorizerProgressBar(input=input_type, max_features=MAX_FEATURES, min_df=MIN_DF, max_df=MAX_DF, stop_words=get_stopwords(), tokenizer=tokenize, ngram_range=(1,N_GRAMS), strip_accents="ascii", dtype=np.uint16)
        count_vectorizer_corpus = count_vectorizer.fit_transform(corpus)
        count_vectorizer.stop_words_ = None  # we can delete this to take up less memory (useful for pickling)
        debug(" -> Done!", 1)

        pickle_dump(count_vectorizer, f"data/{table_name}/pickles/CountVectorizer.pkl")
        pickle_dump(count_vectorizer_corpus, f"data/{table_name}/pickles/CountVectorizer_corpus.pkl")
        with open(f"data/{table_name}/pickles/CountVectorizer_doc_ids.txt", "w") as f:
            for doc_id in doc_ids:
                f.write(doc_id + "\n")


    debug(f" -> Loaded vectorizer with {len(count_vectorizer.get_feature_names())} features!", 1)


    if count_vectorizer_corpus is None and CORPUS_PICKLING and os.path.exists(f"data/{table_name}/pickles/CountVectorizer_corpus.pkl"):
        count_vectorizer_corpus = pickle_load(f"data/{table_name}/pickles/CountVectorizer_corpus.pkl")
        with open(f"data/{table_name}/pickles/CountVectorizer_doc_ids.txt", "r") as f:
            doc_ids = [line[:-1] for line in f]
    elif count_vectorizer_corpus is None:
        debug("Transforming corpus...")
        count_vectorizer_corpus = count_vectorizer.transform(corpus)
        debug(" -> Done!", 1)

        pickle_dump(count_vectorizer_corpus, f"data/{table_name}/pickles/CountVectorizer_corpus.pkl")
        with open(f"data/{table_name}/pickles/CountVectorizer_doc_ids.txt", "w") as f:
            for doc_id in doc_ids:
                f.write(doc_id + "\n")

    debug(f" -> Loaded {count_vectorizer_corpus.shape[0]} documents with {count_vectorizer_corpus.shape[1]} features!", 1)


    return count_vectorizer, doc_ids, count_vectorizer_corpus


def count_vectorize_cache(table_name):

    global n_docs

    corpus_filenames, n_docs = get_cached_filenames(table_name)
    doc_ids = [file[:-4] for file in corpus_filenames]

    count_vectorizer, doc_ids, count_vectorizer_corpus = count_vectorize(doc_ids, corpus_filenames, table_name, input_type="filename")

    return count_vectorizer, doc_ids, count_vectorizer_corpus


def cv_to_tfidf(doc_ids, count_vectorizer_corpus, table_name):

    tfidf_corpus = None

    if MODEL_PICKLING and os.path.exists(f"data/{table_name}/pickles/TfidfTransformer.pkl"):
        tfidf_transformer = pickle_load(f"data/{table_name}/pickles/TfidfTransformer.pkl")

    else:
        debug("Transforming to TF-IDF vector...")
        tfidf_transformer = TfidfTransformer(sublinear_tf=True)
        tfidf_corpus = tfidf_transformer.fit_transform(count_vectorizer_corpus)
        debug(" -> Done!", 1)

        pickle_dump(tfidf_transformer, f"data/{table_name}/pickles/TfidfTransformer.pkl")
        pickle_dump(tfidf_corpus, f"data/{table_name}/pickles/TfidfTransformer_corpus.pkl")
        with open(f"data/{table_name}/pickles/TfidfTransformer_doc_ids.txt", "w") as f:
            for doc_id in doc_ids:
                f.write(doc_id + "\n")


    if tfidf_corpus is None and CORPUS_PICKLING and os.path.exists(f"data/{table_name}/pickles/TfidfTransformer_corpus.pkl"):
        tfidf_corpus = pickle_load(f"data/{table_name}/pickles/TfidfTransformer_corpus.pkl")
        with open(f"data/{table_name}/pickles/TfidfTransformer_doc_ids.txt", "r") as f:
            doc_ids = [line[:-1] for line in f]
    elif tfidf_corpus is None:
        debug("Transforming corpus to TF-IDF...")
        tfidf_corpus = tfidf_transformer.transform(count_vectorizer_corpus)
        debug(" -> Done!", 1)

        pickle_dump(tfidf_corpus, f"data/{table_name}/pickles/TfidfTransformer_corpus.pkl")
        with open(f"data/{table_name}/pickles/TfidfTransformer_doc_ids.txt", "w") as f:
            for doc_id in doc_ids:
                f.write(doc_id + "\n")


    debug(f" -> {tfidf_corpus.shape[0]} vectors transformed!", 1)

    return tfidf_transformer, doc_ids, tfidf_corpus




def main():

    table_name = "fbo_files"

    count_vectorizer, doc_ids, count_vectorizer_corpus = count_vectorize_cache(table_name)
    tfidf_transformer, doc_ids, tfidf_corpus = cv_to_tfidf(doc_ids, count_vectorizer_corpus, table_name)
    print("Done!")

if __name__ == "__main__":
    main()
