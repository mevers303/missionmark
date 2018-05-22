# Mark Evers
# 5/16/18
# nlp.py
# Script for model extraction


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from progress_bar_vetorizers import CountVectorizerProgressBar
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from data import get_cached_filenames, dump_doc_ids, load_doc_ids, get_db_corpus
from globals import *

from pickle_workaround import pickle_dump, pickle_load

import re
from nltk.stem.porter import PorterStemmer as Stemmer

import numpy as np
import os




def get_stopwords():
    """
    Reads the stopwords from stopwords.txt and combines them with the ENGLISH_STOP_WORDS in sklearn.
    :return: A set of stopwords.
    """

    with open("../stopwords.txt", "r") as f:
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



def get_cached_corpus(table_name, name):
    
    if os.path.exists(f"../data/{table_name}/pickles/{name}_corpus.pkl") and os.path.exists(f"../data/{table_name}/pickles/{name}_doc_ids.txt"):
        doc_ids = load_doc_ids(f"../data/{table_name}/pickles/{name}_doc_ids.txt")
        cv_corpus = pickle_load(f"../data/{table_name}/pickles/{name}_corpus.pkl")
        return doc_ids, cv_corpus
    else:
        return None, None




def cache_corpus(doc_ids, corpus, table_name, name):

    dump_doc_ids(doc_ids, f"../data/{table_name}/pickles/{name}_corpus_doc_ids.txt")
    pickle_dump(corpus, f"../data/{table_name}/pickles/{name}_corpus.pkl")




def count_vectorize(corpus, table_name, model_from_pickle, input_type="content"):

    cv_corpus = None


    if model_from_pickle and os.path.exists(f"../data/{table_name}/pickles/CountVectorizer.pkl"):
        count_vectorizer = pickle_load(f"../data/{table_name}/pickles/CountVectorizer.pkl")

    else:
        debug("Vectorizing documents...")
        count_vectorizer = CountVectorizerProgressBar(input=input_type, max_features=MAX_FEATURES, min_df=MIN_DF, max_df=MAX_DF, stop_words=get_stopwords(), tokenizer=tokenize, ngram_range=(1, N_GRAMS), strip_accents="ascii", dtype=np.uint16, progress_bar_clear=True)
        cv_corpus = count_vectorizer.fit_transform(corpus)
        count_vectorizer.stop_words_ = None  # we can delete this to take up less memory (useful for pickling)
        debug(" -> Done!", 1)

    debug(f" -> Loaded vectorizer with {len(count_vectorizer.get_feature_names())} features!", 1)


    if cv_corpus is None:
        debug("Transforming corpus...")
        cv_corpus = count_vectorizer.transform(corpus)
        debug(" -> Done!", 1)


    debug(f" -> Loaded {cv_corpus.shape[0]} documents with {cv_corpus.shape[1]} features!", 1)
    return count_vectorizer, cv_corpus


def count_vectorize_cache(table_name):

    corpus_filenames, n_docs = get_cached_filenames(table_name)
    doc_ids = [file[:-4] for file in corpus_filenames]

    count_vectorizer, doc_ids, cv_corpus = count_vectorize(doc_ids, corpus_filenames, table_name, input_type="filename")

    return count_vectorizer, doc_ids, cv_corpus


def cv_to_tfidf(cv_corpus, table_name, model_from_pickle):

    tfidf_corpus = None


    if model_from_pickle and os.path.exists(f"../data/{table_name}/pickles/TfidfTransformer.pkl"):
        tfidf_transformer = pickle_load(f"../data/{table_name}/pickles/TfidfTransformer.pkl")

    else:
        debug("Transforming to TF-IDF vector...")
        tfidf_transformer = TfidfTransformer(sublinear_tf=True)
        tfidf_corpus = tfidf_transformer.fit_transform(cv_corpus)
        debug(" -> Done!", 1)


    if tfidf_corpus is None:
        debug("Transforming corpus to TF-IDF...")
        tfidf_corpus = tfidf_transformer.transform(cv_corpus)
        debug(" -> Done!", 1)


    debug(f" -> {tfidf_corpus.shape[0]} count vectors with {tfidf_corpus.shape[1]} features transformed!", 1)
    return tfidf_transformer, tfidf_corpus




def build_model_and_corpus_cache(doc_ids, corpus, table_name, input_type="content"):

    cv_model, cv_corpus = count_vectorize(corpus, table_name, False, input_type)
    pickle_dump(cv_model, f"../data/{table_name}/pickles/CountVectorizer.pkl")
    cache_corpus(doc_ids, cv_corpus, table_name, "cv")

    del cv_model  # save some memory
    tfidf_model, tfidf_corpus = cv_to_tfidf(cv_corpus, table_name, False)
    del cv_corpus  # save some memory
    pickle_dump(tfidf_model, f"../data/{table_name}/pickles/TfidfTransformer.pkl")
    cache_corpus(doc_ids, tfidf_corpus, table_name, "tfidf")

    print("Done!")



def main():

    get_command_line_options()
    doc_ids, corpus = get_db_corpus(TABLE_NAME, ID_COLUMN, TEXT_COLUMN, remove_html=STRIP_HTML)
    build_model_and_corpus_cache(doc_ids, corpus, TABLE_NAME)


if __name__ == "__main__":
    main()
