# Mark Evers
# 5/7/18
# topics.py
# Script for model extraction


from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from src.data import *

import re
import os
from nltk.stem.porter import PorterStemmer

import numpy as np


stemmer = PorterStemmer()
def tokenize(text):
    """Tokenizes a document."""
    return [stemmer.stem(token) for token in split_tokens_hard(text)]


def print_top_words(H, words, num_words=100):

    word_indices = np.array(np.argsort(H, axis=1)[:, :-num_words - 1:-1])
    top_words = []

    for topic_i in range(word_indices.shape[0]):
        top_words.append([words[word] for word in word_indices[topic_i]])

    for topic_i in range(len(top_words)):

        other_words = set()
        for i in range(len(top_words)):
            if i == topic_i:
                continue
            other_words.update(top_words[i])

        unique_words = set(top_words[topic_i]) - other_words

        print(f"Top words for latent feature {topic_i}:\n", unique_words)
        print() # newline



def print_top_docs(W, titles):

    indexes = np.argsort(W, axis=0)[:-11:-1, :]

    for column in range(indexes.shape[1]):
        print("Top articles for topic {0}:".format(column))
        for i in indexes[:, column]:
            print(titles[i])
        print()




def get_stopwords():

    with open("stopwords.txt", "r") as f:
        custom_stopwords = {word for word in f.readline()}

    return ENGLISH_STOP_WORDS.union(custom_stopwords)






vectorizer = TfidfVectorizer(stop_words=get_stopwords(), tokenizer=tokenize, max_df=.75)
corpus = get_test_data()
corpus_tfidf = vectorizer.fit_transform(corpus)

model = NMF(n_components=25, max_iter=100)

W = model.fit_transform(corpus_tfidf)
H = model.components_



summaries = []
for doc in corpus:

    sentences = split_sentences(doc)
    sentence_tfidf = vectorizer.transform(sentences).toarray()
    # sentences_wordcounts = np.count_nonzero(sentence_tfidf, axis=1)
    sentence_scores = np.sum(sentence_tfidf, axis=1).flatten()  # / sentences_wordcounts
    best_sentences = [f"{'*' * 120}\n{sentences[i]}." for i in np.sort(np.argsort(sentence_scores)[:-11:-1])]
    summaries.append("\n\n\n".join(best_sentences))




topics = np.argmax(W, axis=1)
for i in range(topics.size):

    path = os.path.join("output", str(topics[i]).rjust(2, "0"))
    if not os.path.isdir(path):
        os.mkdir(path)

    filename = os.path.join(path, f"{i}.txt".rjust(7, "0"))
    with open(filename, "w") as f:
        f.write(summaries[i])
