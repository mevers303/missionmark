# Mark Evers
# 5/7/18
# nlp.py
# Script for model extraction


from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from src.data import *
from src.globals import *

import re
import os
from nltk.stem.porter import PorterStemmer

import numpy as np


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



stemmer = PorterStemmer()
def tfidf_tokenize(text):
    """
    Tokenizes a document to be converted to a TF-IDF vector.
    :param text: The text/document to be tokenized.
    :return: A list of stemmed tokens.
    """
    return [stemmer.stem(token) for token in split_tokens_hard(text)]



def get_top_topic_words(H):
    """
    Gets a matrix of the most relevent word indices (columns) for each topic (rows).
    :param H: The H matrix from NMF.
    :return: A matrix of the most relevent word indices (columns) for each topic (rows).
    """
    return np.argsort(H, axis=1)[:, ::-1]


def print_top_topic_words(H, vectorizer, n_words=100):
    """
    Prints the top unique words for predicting a topic.
    :param H: The H matrix from NMF.
    :param vectorizer: The TF-IDF vectorizer to get the word list from.
    :param num_words: Total number of words to look at.
    :return: None
    """

    word_list = vectorizer.get_feature_names()
    top_word_indices = get_top_topic_words(H)[:, :-n_words - 1:-1]
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
        custom_stopwords = {word for word in f.readline()}

    return ENGLISH_STOP_WORDS.union(custom_stopwords)



def summarize_doc(doc, vectorizer, n_sentences = 10):
    """
    Auto summarizes a document.
    :param doc: The document to be summarized.
    :param vectorizer: The TF-IDF vectorizer to be used for feature extraction.
    :param n_sentences: Number of sentences to include in the summary.
    :return: A string containing the best sentences, in order.
    """

    sentences = split_sentences(doc)
    sentence_tfidf = vectorizer.transform(sentences).toarray()
    # sentences_wordcounts = np.count_nonzero(sentence_tfidf, axis=1)

    sentence_scores = np.sum(sentence_tfidf, axis=1).flatten()  # / sentences_wordcounts
    best_sentences = [f"{'*' * 120}\n{'*' * 120}\n{'*' * 120}\n{sentences[i]}." for i in np.sort(np.argsort(sentence_scores)[:-n_sentences - 1:-1])]

    return "\n\n\n".join(best_sentences)



def sumarize_corpus(corpus, n_sentences=10):
    """
    Summarizes an entire corpus.  Displays a progress bar.
    :param corpus: The corpus to be summarized
    :param n_sentences: Number of sentences to include in the summary.
    :return: A corpus of summaries
    """

    summaries = []
    n_docs = len(corpus)
    completed = 0

    for doc in corpus:
        summaries.append(summarize_doc(doc, vectorizer, n_sentences))
        completed += 1
        progress_bar(completed, n_docs)

    return summaries





def dump_topic_corpus(W, corpus):
    """
    Saves the corpus to output/<topic #>/<filename>.txt, organized by topic.
    :param W: The W matrix from NMF.
    :param corpus: The corpus to be saved
    :return: None
    """

    doc_topics = np.argmax(W, axis=1)

    for i in range(doc_topics.size):

        path = os.path.join("output", str(doc_topics[i]).rjust(2, "0"))
        if not os.path.isdir(path):
            os.mkdir(path)

        filename = os.path.join(path, f"{i}.txt".rjust(7, "0"))
        with open(filename, "w") as f:
            f.write(corpus[i])



# def main():
if __name__ == "__main__":

    n_topics = 25

    print("Loading corpus...")
    corpus = get_test_data()
    print(f" -> {len(corpus)} documents loaded!\n")

    print("Vectorizing keywords...")
    vectorizer = TfidfVectorizer(stop_words=get_stopwords(), tokenizer=tfidf_tokenize, max_df=.75)
    corpus_tfidf = vectorizer.fit_transform(corpus)
    print(f" -> {corpus_tfidf.shape[1]} tokens found!\n")

    print(f"Sorting into {n_topics} topics...")
    model = NMF(n_components=25, max_iter=500)
    W = model.fit_transform(corpus_tfidf)
    H = model.components_
    print(f" -> {model.n_iter_} iterations completed!\n")

    print("Summarizing documents...")
    summaries = sumarize_corpus(corpus)
    print(f" -> {len(summaries)} documents summarized!\n")
    print("Saving summaries to disk based on topic...")
    dump_topic_corpus(W, summaries)
    print(f" -> {len(summaries)} files created!\n")

    print_top_topic_words(H, vectorizer)
    with open("features.txt", "w") as f:
        for word in vectorizer.get_feature_names():
            f.write(word + "\n")

    print("Done!")



# if __name__ == "__main__":
#     main()
