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


stemmer = PorterStemmer()
def tfidf_tokenize(text):
    """Tokenizes a document."""
    return [stemmer.stem(token) for token in split_tokens_hard(text)]



def get_top_topic_words(H, words, num_words=100):

    word_indices = np.argsort(H, axis=1)[:, :-num_words - 1:-1]
    top_words = [[words[word] for word in topic] for topic in word_indices]

    for topic in top_words:
        other_words = {word for other_topic in top_words if other_topic != topic for word in other_topic}
        unique_words = set(topic) - other_words

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


def sumarize_corpus(corpus):

    summaries = []
    n_docs = len(corpus)
    completed = 0

    for doc in corpus:
        summaries.append(summarize_doc(doc, vectorizer))
        completed += 1
        progress_bar(completed, n_docs)

    return summaries





def dump_topic_summaries(W, summaries):

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
    dump_topic_summaries(W, summaries)
    print(f" -> {len(summaries)} files created!\n")

    print("Done!")



# if __name__ == "__main__":
#     main()
