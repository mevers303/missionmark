# Mark Evers
# 5/21/18


import globals as g


import re
import numpy as np


def split_sentences(doc):
    """
    Splits a document into sentences.
    :param doc: The document to be split
    :return: A list of sentences.
    """
    return re.split(r"[\.\?\!]\s+", doc)



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

    g.debug("Summarizing documents...")

    summaries = []
    n_docs = len(corpus)
    completed = 0

    for doc in corpus:
        summaries.append(summarize_doc(doc, vectorizer, n_sentences))
        completed += 1
        g.progress_bar(completed, n_docs, 1)

    g.debug(f" -> {len(summaries)} documents summarized!", 1)
    return summaries