# Mark Evers
# 5/7/18
# nlp.py
# Script for model extraction


from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from src.data import *
from src.globals import *

import pickle

import re
import os
from nltk.stem.wordnet import WordNetLemmatizer as Stemmer

import numpy as np

from wordcloud import WordCloud





stemmer = Stemmer()
def tfidf_tokenize(text):
    """
    Tokenizes a document to be converted to a TF-IDF vector.
    :param text: The text/document to be tokenized.
    :return: A list of stemmed tokens.
    """
    return [stem(token) for token in split_tokens_hard(text)]
    # return [token for token in split_tokens_hard(text)]


def stem(token):

    if token.endswith("ies") and len(token) > 6:
        return token.replace("ies", "y")

    if token.endswith("s") and len(token) > 5 and not token.endswith("ss"):
        return token[:-1]

    # if it's too long, it's probably a mashup of words, return a stopword
    if len(token) > 16:
        return "a"

    return token


def pickle_save(doc_ids, corpus, vectorizer, corpus_tfidf, model, W):

    save_corpus_pickle(doc_ids, corpus)

    if PICKLING:
        return

    with open("pickle/tfidf.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open("pickle/corpus_tfidf.pkl", "wb") as f:
        pickle.dump(corpus_tfidf, f)

    with open("pickle/nmf.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("pickle/W.pkl", "wb") as f:
        pickle.dump(W, f)


def split_sentences(doc):
    """
    Splits a document into sentences.
    :param doc: The document to be split
    :return: A list of sentences.
    """
    return re.split(r"[\.\?\!]\s+", doc)


def split_tokens_hard(text):
    """
    Splits into tokens based on any character that is NOT a letter, number, "-", or ".".
    :param text: The text to be tokenized
    :return: A list of tokens
    """
    # return [token for token in re.split(r"[^a-zA-Z0-9\-\.]+|[\-\.]{3,}|\s[\-\.]+|[\-\.]+\s", text) if token]  # list comprehension removes empty strings
    # return [token for token in re.split(r"[^a-zA-Z0-9]+", text) if token]  # list comprehension removes empty strings
    return [token for token in re.split(r"[^a-zA-Z]+", text) if token]  # list comprehension removes empty strings


def split_tokens_soft(text):
    """
    Splits into tokens, still allowing for special characters (for acronym extraction).
    :param text: The text to be tokenized.
    :return: A list of tokens.
    """
    return [token for token in re.split(r"[\s/\-\+,\\\"\:\;\[\]]+", text) if token]  # list comprehension removes empty strings



def get_top_topic_words(H):
    """
    Gets a matrix of the most relevent word indices (columns) for each topic (rows).
    :param H: The H matrix from NMF.
    :return: A matrix of the most relevent word indices (columns) for each topic (rows).
    """
    return np.argsort(H, axis=1)[:, ::-1]



def print_top_topic_words(H, word_list, n_words=100):
    """
    Prints the top unique words for predicting a topic.
    :param H: The H matrix from NMF.
    :param vectorizer: The TF-IDF vectorizer to get the word list from.
    :param num_words: Total number of words to look at.
    :return: None
    """

    top_word_indices = get_top_topic_words(H)[:, :n_words]
    top_words = [[word_list[word] for word in topic] for topic in top_word_indices]

    topic_number = 0
    for topic in top_words:
        other_words = {word for other_topic in top_words if other_topic != topic for word in other_topic}
        unique_words = set(topic) - other_words

        topic_number += 1
        print(f"Top words for latent feature {topic_number}:\n", unique_words)
        print() # newline




def get_stopwords():
    """
    Reads the stopwords from stopwords.txt and combines them with the ENGLISH_STOP_WORDS in sklearn.
    :return: A set of stopwords.
    """

    with open("stopwords.txt", "r") as f:
        custom_stopwords = {word[:-1] for word in f}

    return list(ENGLISH_STOP_WORDS.union(custom_stopwords))



def summarize_doc(doc, vectorizer, n_sentences=10):
    """
    Auto summarizes a document.
    :param doc: The document to be summarized.
    :param vectorizer: The TF-IDF vectorizer to be used for feature extraction.
    :param n_sentences: Number of sentences to include in the summary.
    :return: A string containing the best sentences, in order.
    """

    sentences = split_sentences(doc)
    sentence_tfidf = vectorizer.transform(sentences)
    # sentences_wordcounts = np.count_nonzero(sentence_tfidf, axis=1)

    sentence_scores = sentence_tfidf.sum(axis=1).A1  # / sentences_wordcounts
    best_sentences_i = np.sort(np.argsort(sentence_scores)[:-n_sentences - 1:-1])
    best_sentences = [f"{'*' * 120}\n{'*' * 120}\n{'*' * 120}\n{sentences[i]}." for i in best_sentences_i]

    return "\n\n\n".join(best_sentences)



def sumarize_corpus(corpus, vectorizer, n_sentences=10):
    """
    Summarizes an entire corpus.  Displays a progress bar.
    :param corpus: The corpus to be summarized
    :param n_sentences: Number of sentences to include in the summary.
    :return: A corpus of summaries
    """

    debug("Summarizing documents...")

    summaries = []
    n_docs = len(corpus)
    completed = 0

    for doc in corpus:
        summaries.append(summarize_doc(doc, vectorizer, n_sentences))
        completed += 1
        progress_bar(completed, n_docs, 1)

    debug(f" -> {len(summaries)} documents summarized!", 1)
    return summaries



def get_corpus_top_topics(W):
    """
    Gets the best single topic for each document.
    :param W: W from NMF
    :return: An array of topic indexes.
    """
    return np.argmax(W, axis=1)




def get_corpus_topic_strengths(W):
    """
    Gets a matrix of topic strengths
    :param W:
    :return:
    """
    return np.argsort(W, axis=1)[:, ::-1]




def dump_topic_corpus(corpus_topics, corpus, doc_ids):
    """
    Saves the corpus to output/<topic #>/<filename>.txt, organized by topic.
    :param W: The W matrix from NMF.
    :param corpus: The corpus to be saved
    :return: None
    """

    debug("Saving summaries to disk based on topic...")

    for i in range(corpus_topics.size):

        path = os.path.join("output", str(corpus_topics[i]).rjust(2, "0"))
        if not os.path.isdir(path):
            os.mkdir(path)

        filename = os.path.join(path, f"{doc_ids[i]}.txt")
        with open(filename, "w") as f:
            f.write(corpus[i])

    debug(f" -> {len(corpus)} files created!", 1)


def get_tfidf_topic_weights(corpus_tfidf, corpus_topics, n_topics):

    topic_tfidf_weights = []

    for topic_i in range(n_topics):

        topic_corpus = corpus_tfidf[corpus_topics == topic_i]

        topic_word_scores = topic_corpus.sum(axis=0).A1
        topic_tfidf_weights.append(topic_word_scores)

    return np.array(topic_tfidf_weights)



def get_tfidf_topic_words(corpus_tfidf, corpus_topics, word_list, n_topics, n_words=10):

    topic_tfidf_weights = get_tfidf_topic_weights(corpus_tfidf, corpus_topics, n_topics)
    topic_top_words_i = np.argsort(topic_tfidf_weights, axis=1)[:, ::-1]
    top_words = []

    for topic_i in range(n_topics):

        topic_words = word_list[topic_top_words_i[topic_i, :n_words]]
        top_words.append(topic_words)

        debug(f"TF-IDF words for topic {topic_i}:")
        debug(str(topic_words))

    return top_words



def build_word_clouds(corpus_tfidf, corpus_topics, topic_nmf_weights, word_list, n_topics):

    debug("Generating topic word clouds... (this may take a while)")
    completed = 0
    progress_bar(completed, n_topics)

    topic_tfidf_weights = get_tfidf_topic_weights(corpus_tfidf, corpus_topics, n_topics)
    topic_top_tfidf_words_i = np.argsort(topic_tfidf_weights, axis=1)[:, ::-1]
    topic_top_nmf_words_i = np.argsort(topic_nmf_weights, axis=1)[:, ::-1]

    for topic_i in range(n_topics):

        path = os.path.join("output", str(topic_i).rjust(2, "0"))
        if not os.path.isdir(path):
            os.mkdir(path)

        # nmf wordcloud
        wc = WordCloud(background_color="black", max_words=666, width=2000, height=1000)
        wc.fit_words({word_list[word_i]: topic_nmf_weights[topic_i, word_i] for word_i in topic_top_nmf_words_i[topic_i] if topic_nmf_weights[topic_i, word_i]})
        wc.to_file(os.path.join(path, "nmf_wordcloud.png"))

        # an empty topic...
        if not topic_tfidf_weights[topic_i].sum():
            continue

        # tf-idf wordcloud
        wc = WordCloud(background_color="black", max_words=666, width=2000, height=1000)
        wc.fit_words({word_list[word_i]: topic_tfidf_weights[topic_i, word_i] for word_i in topic_top_tfidf_words_i[topic_i] if topic_tfidf_weights[topic_i, word_i]})
        wc.to_file(os.path.join(path, "tf-idf_wordcloud.png"))

        completed += 1
        progress_bar(completed, n_topics)

    debug(f" -> {n_topics} word clouds generated!")



def vectorize(corpus):

    debug("Vectorizing keywords...")

    if PICKLING:
        debug(" -> Loading cached corpus vector...")

        with open("pickle/tfidf.pkl", "rb") as f:
            vectorizer = pickle.load(f)

        with open("pickle/corpus_tfidf.pkl", "rb") as f:
            corpus_tfidf = pickle.load(f)

    else:
        vectorizer = TfidfVectorizer(stop_words=get_stopwords(), tokenizer=tfidf_tokenize, max_df=.66, min_df=2, ngram_range=(1,1), sublinear_tf=True)
        corpus_tfidf = vectorizer.fit_transform(corpus)

    debug(f" -> {corpus_tfidf.shape[1]} tokens found!", 1)
    return vectorizer, corpus_tfidf



def dump_features(word_list):

    debug("Writing word list to features.txt...")

    with open("features.txt", "w") as f:
        for word in word_list:
            f.write(word + "\n")

    debug(f" -> Wrote {len(word_list)} to file!")




def nmf_model(corpus_tfidf, n_topics, max_iter=500):

    debug(f"Sorting into {n_topics} topics...")

    if PICKLING:
        debug(" -> Loading cached model...")
        with open("pickle/nmf.pkl", "rb") as f:
            model = pickle.load(f)

        with open("pickle/W.pkl", "rb") as f:
            W = pickle.load(f)

        H = model.components_

    else:
        model = NMF(n_components=n_topics, max_iter=max_iter)
        W = model.fit_transform(corpus_tfidf)
        H = model.components_

    debug(f" -> {model.n_iter_} iterations completed!", 1)
    return model, W, H


def main():
# if __name__ == "__main__":

    DEBUG_LEVEL = 2
    n_topics = 100
    do_summaries = False

    doc_ids, corpus = get_corpus()

    vectorizer, corpus_tfidf = vectorize(corpus)
    word_list = np.array(vectorizer.get_feature_names())

    if DEBUG_LEVEL > 1:
        dump_features(word_list)
    # exit(0)

    model, W, H = nmf_model(corpus_tfidf, n_topics)
    corpus_topics = get_corpus_top_topics(W)

    pickle_save(doc_ids, corpus, vectorizer, corpus_tfidf, model, W)

    if do_summaries:
        summaries = sumarize_corpus(corpus, vectorizer)
        dump_topic_corpus(corpus_topics, summaries, doc_ids)
        del summaries  # save some RAM for the wordclouds

    if DEBUG_LEVEL > 1:
         print_top_topic_words(H, word_list, 50)
         print(get_tfidf_topic_words(corpus_tfidf, corpus_topics, word_list, n_topics, 15))

    build_word_clouds(corpus_tfidf, corpus_topics, H, word_list, n_topics)


    debug("Done!")


if __name__ == "__main__":
    main()
