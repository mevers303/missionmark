# Mark Evers
# 5/7/18
# nlp.py
# Script for model extraction


from sklearn.decomposition import NMF
import globals as g
from vectorizer import get_cached_corpus, get_features

from pickle_workaround import pickle_dump, pickle_load


import re
import os

import numpy as np

from wordcloud import WordCloud





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


def get_top_10_docs(W, topic):
    """
    Returns the top 10 documents for a topic
    :param W: W from NMF.
    :param topic: The topic index.
    :return: List of tuples, the index of the best documents and their strength.
    """

    docs = np.argsort(W[:, topic].flatten())[::-1][:10]
    strengths = W[:, topic].flatten()[docs]

    return [(doc, strength) for doc, strength in zip(docs, strengths)]




def dump_topic_corpus(corpus_topics, corpus, doc_ids):
    """
    Saves the corpus to output/<topic #>/<filename>.txt, organized by topic.
    :param W: The W matrix from NMF.
    :param corpus: The corpus to be saved
    :return: None
    """

    g.debug("Saving summaries to disk based on topic...")

    for i in range(corpus_topics.size):

        path = os.path.join("output", str(corpus_topics[i]).rjust(2, "0"))
        if not os.path.isdir(path):
            os.mkdir(path)

        filename = os.path.join(path, f"{doc_ids[i]}.txt")
        with open(filename, "w") as f:
            f.write(corpus[i])

    g.debug(f" -> {len(corpus)} files created!", 1)


def get_tfidf_topic_weights(corpus_tfidf, corpus_topics, n_topics):

    topic_tfidf_weights = []

    for topic_i in range(n_topics):

        topic_corpus = corpus_tfidf[corpus_topics == topic_i]

        topic_word_scores = topic_corpus.sum(axis=0).A1
        topic_tfidf_weights.append(topic_word_scores)

    return np.array(topic_tfidf_weights)



def print_tfidf_topic_words(corpus_tfidf, corpus_topics, word_list, n_topics, n_words=10):

    topic_tfidf_weights = get_tfidf_topic_weights(corpus_tfidf, corpus_topics, n_topics)
    topic_top_words_i = np.argsort(topic_tfidf_weights, axis=1)[:, ::-1]
    top_words = []

    for topic_i in range(n_topics):

        topic_words = word_list[topic_top_words_i[topic_i, :n_words]]
        top_words.append(topic_words)

        g.debug(f"TF-IDF words for topic {topic_i}:", 2)
        g.debug(str(topic_words), 2)

    return top_words



def build_word_clouds(corpus_tfidf, corpus_topics, H, word_list, table_name):

    g.debug("Generating topic word clouds...")
    n_topics = H.shape[0]
    completed = 0
    g.progress_bar(completed, n_topics)

    topic_tfidf_weights = get_tfidf_topic_weights(corpus_tfidf, corpus_topics, n_topics)
    topic_top_tfidf_words_i = np.argsort(topic_tfidf_weights, axis=1)[:, ::-1]
    topic_top_nmf_words_i = np.argsort(H, axis=1)[:, ::-1]

    for topic_i in range(n_topics):

        # nmf wordcloud
        wc = WordCloud(background_color="black", max_words=666, width=2000, height=1000)
        wc.fit_words({word_list[word_i]: H[topic_i, word_i] for word_i in topic_top_nmf_words_i[topic_i] if H[topic_i, word_i]})
        wc.to_file(f"../output/{table_name}/nmf/{topic_i}_nmf_wordcloud.png")

        # an empty topic...
        if not topic_tfidf_weights[topic_i].sum():
            continue

        # tf-idf wordcloud
        wc = WordCloud(background_color="black", max_words=666, width=2000, height=1000)
        wc.fit_words({word_list[word_i]: topic_tfidf_weights[topic_i, word_i] for word_i in topic_top_tfidf_words_i[topic_i] if topic_tfidf_weights[topic_i, word_i]})
        wc.to_file(f"../output/{table_name}/nmf/{topic_i}_tfidf_wordcloud.png")

        completed += 1
        g.progress_bar(completed, n_topics)

    g.debug(f" -> {n_topics} word clouds generated!", 1)




def nmf_model(corpus_tfidf, n_topics, max_iter=500, no_output=False):

    if not no_output:
        g.debug(f"Sorting corpus into {n_topics} topics...")

    nmf = NMF(n_components=n_topics, max_iter=max_iter, random_state=666)
    W = nmf.fit_transform(corpus_tfidf)
    H = nmf.components_

    if not no_output:
        g.debug(f" -> {nmf.n_iter_} iterations completed!", 1)

    return nmf, W, H



def build_model_and_wordclouds(n_topics, tfidf_corpus, vocabulary, table_name):

    nmf, W, H = nmf_model(tfidf_corpus, n_topics, max_iter=666)
    pickle_dump(nmf, f"../data/{table_name}/pickles/NMF.pkl")
    pickle_dump(W, f"../data/{table_name}/pickles/W.pkl")

    corpus_topics = get_corpus_top_topics(W)
    build_word_clouds(tfidf_corpus, corpus_topics, H, vocabulary, table_name)

    g.debug("Done!")



def main():

    g.get_command_line_options()

    doc_ids, tfidf_corpus = get_cached_corpus(g.TABLE_NAME, "tfidf")
    vocabulary = get_features(g.TABLE_NAME)

    build_model_and_wordclouds(g.N_TOPICS, tfidf_corpus, vocabulary, g.TABLE_NAME)


if __name__ == "__main__":
    main()
