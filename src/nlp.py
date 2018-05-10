# Mark Evers
# 5/7/18
# nlp.py
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
def tfidf_tokenize(text):
    """Tokenizes a document."""
    return [stemmer.stem(token) for token in split_tokens_hard(text)]



def get_top_words(H, words, num_words=100):

    word_indices = np.argsort(H, axis=1)[:, :-num_words - 1:-1]
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



def summarize_doc(doc, vectorizer):

    sentences = split_sentences(doc)
    sentence_tfidf = vectorizer.transform(sentences).toarray()
    # sentences_wordcounts = np.count_nonzero(sentence_tfidf, axis=1)

    sentence_scores = np.sum(sentence_tfidf, axis=1).flatten()  # / sentences_wordcounts
    best_sentences = [f"{'*' * 120}\n{'*' * 120}\n{'*' * 120}\n{sentences[i]}." for i in np.sort(np.argsort(sentence_scores)[:-11:-1])]

    return "\n\n\n".join(best_sentences)



def dump_topic_summaries(W, summaries):

    print("Dumping summaries to file based on topic...")
    doc_topics = np.argmax(W, axis=1)

    for i in range(doc_topics.size):

        path = os.path.join("output", str(doc_topics[i]).rjust(2, "0"))
        if not os.path.isdir(path):
            os.mkdir(path)

        filename = os.path.join(path, f"{i}.txt".rjust(7, "0"))
        with open(filename, "w") as f:
            f.write(summaries[i])



# def main():
if __name__ == "__main__":

    vectorizer = TfidfVectorizer(stop_words=get_stopwords(), tokenizer=tfidf_tokenize, max_df=.75)
    corpus = get_test_data()

    print("Vectorizing keywords...")
    corpus_tfidf = vectorizer.fit_transform(corpus)

    print("Searching for latent topics...")
    model = NMF(n_components=25, max_iter=100)
    W = model.fit_transform(corpus_tfidf)
    H = model.components_

    print("Summarizing documents...")
    summaries = [summarize_doc(doc, vectorizer) for doc in corpus]
    dump_topic_summaries(W, summaries)

    print("Done!")



# if __name__ == "__main__":
#     main()
