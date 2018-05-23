# Mark Evers
# 5/16/18
# nlp.py
# Script for model extraction


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from progress_bar_vetorizers import CountVectorizerProgressBar
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from data import get_cached_filenames, dump_doc_ids, load_doc_ids, get_db_corpus
import globals as g

from pickle_workaround import pickle_dump, pickle_load

import re
from nltk.stem.porter import PorterStemmer as Stemmer

import numpy as np
import os
import pandas as pd




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




def cache_corpus(corpus_df, table_name, name):

    pickle_dump(corpus_df, f"../data/{table_name}/pickles/{name}_corpus.pkl")




def count_vectorize(corpus_df, table_name, model_from_pickle, input_type="content"):
    # TODO find all usages
    cv_corpus_df = None


    if model_from_pickle and os.path.exists(f"../data/{table_name}/pickles/CountVectorizer.pkl"):
        count_vectorizer = pickle_load(f"../data/{table_name}/pickles/CountVectorizer.pkl")

    else:
        g.debug("Vectorizing documents...")
        count_vectorizer = CountVectorizerProgressBar(input=input_type, max_features=g.MAX_FEATURES, min_df=g.MIN_DF, max_df=g.MAX_DF, stop_words=get_stopwords(), tokenizer=tokenize, ngram_range=(1, g.N_GRAMS), strip_accents="ascii", dtype=np.uint16, progress_bar_clear=True)
        cv_corpus_df = pd.DataFrame(data=count_vectorizer.fit_transform(corpus_df.values[:, 0]), index=corpus_df.index)
        cv_corpus_df.columns = count_vectorizer.get_feature_names()
        count_vectorizer.stop_words_ = None  # we can delete this to take up less memory (useful for pickling)
        g.debug(" -> Done!", 1)

    g.debug(f" -> Loaded vectorizer with {len(count_vectorizer.get_feature_names())} features!", 1)


    if cv_corpus_df is None:
        g.debug("Transforming corpus...")
        cv_corpus_df = pd.DataFrame(data=count_vectorizer.transform(corpus_df.values[:, 0]), index=corpus_df.index, columns=count_vectorizer.get_feature_names())
        g.debug(" -> Done!", 1)


    g.debug(f" -> Loaded {cv_corpus_df.shape[0]} documents with {cv_corpus_df.shape[1]} features!", 1)
    return count_vectorizer, cv_corpus_df



def cv_to_tfidf(cv_corpus_df, table_name, model_from_pickle):
    # TODO find all usages
    tfidf_corpus_df = None


    if model_from_pickle and os.path.exists(f"../data/{table_name}/pickles/TfidfTransformer.pkl"):
        tfidf_transformer = pickle_load(f"../data/{table_name}/pickles/TfidfTransformer.pkl")

    else:
        g.debug("Transforming to TF-IDF vector...")
        tfidf_transformer = TfidfTransformer(sublinear_tf=True)
        tfidf_corpus_df = pd.DataFrame(data=tfidf_transformer.fit_transform(cv_corpus_df.values), index=cv_corpus_df.index, columns=cv_corpus_df.columns)
        g.debug(" -> Done!", 1)


    if tfidf_corpus_df is None:
        g.debug("Transforming corpus to TF-IDF...")
        tfidf_corpus_df = pd.DataFrame(data=tfidf_transformer.transform(cv_corpus_df.values), index=cv_corpus_df.index, columns=cv_corpus_df.columns)
        g.debug(" -> Done!", 1)


    g.debug(f" -> {tfidf_corpus_df.shape[0]} vectors with {tfidf_corpus_df.shape[1]} features transformed to TF-IDF!", 1)
    return tfidf_transformer, tfidf_corpus_df



def tfidf_vectorize(corpus, table_name, model_from_pickle, input_type="content"):

    count_vectorizer, cv_corpus = count_vectorize(corpus, table_name, model_from_pickle, input_type)
    vocabulary = count_vectorizer.get_feature_names()
    _, tfidf_corpus = cv_to_tfidf(cv_corpus, table_name, model_from_pickle)

    return tfidf_corpus, vocabulary





def dump_features(word_list, table_name):

    g.debug("Writing word list to features.txt...")

    with open(f"../data/{table_name}/pickles/features.txt", "w") as f:
        for word in word_list:
            f.write(word + "\n")

    g.debug(f" -> Wrote {len(word_list)} to file!", 1)



def get_features(table_name):

    return [word[:-1] for word in open(f"../data/{table_name}/pickles/features.txt", "r")]







def build_model_and_corpus_cache(corpus_df, table_name, input_type="content"):

    cv_model, cv_corpus_df = count_vectorize(corpus_df, table_name, False, input_type)
    pickle_dump(cv_model, f"../data/{table_name}/pickles/CountVectorizer.pkl")
    dump_features(cv_model.get_feature_names(), table_name)
    cache_corpus(cv_corpus_df, table_name, "cv")

    del cv_model  # save some memory
    tfidf_model, tfidf_corpus = cv_to_tfidf(cv_corpus_df, table_name, False)
    del cv_corpus_df  # save some memory
    pickle_dump(tfidf_model, f"../data/{table_name}/pickles/TfidfTransformer.pkl")
    cache_corpus(tfidf_corpus, table_name, "tfidf")

    print("Done!")



def main():

    g.get_command_line_options()

    corpus_df = get_db_corpus(g.TABLE_NAME, g.ID_COLUMN, g.TEXT_COLUMN, remove_html=g.STRIP_HTML)
    build_model_and_corpus_cache(corpus_df, g.TABLE_NAME)


if __name__ == "__main__":
    main()
