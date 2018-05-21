# Mark Evers
#  5/20/18


import os
import sys
sys.path.append(os.getcwd())
sys.path.append("src")

import numpy as np

from src.nlp import nmf_model
from src.globals import *
from src.vectorizer import count_vectorize_cache, cv_to_tfidf, count_vectorize
from src.data import get_db_corpus


def rss_cost(V, W, H):
    return (np.array(V - W.dot(H))**2).sum()


def search_models(tfidf_corpus, min_topics, max_topics):

    debug("Building NMF topics...")
    nmf_models = []
    costs = []

    progress_bar(0, max_topics)
    for i in range(min_topics, max_topics + 1):

        nmf, W, H = nmf_model(tfidf_corpus, i, skip_pickling=True)
        rss = rss_cost(tfidf_corpus, W, H)

        nmf_models.append(nmf)
        costs.append(rss)
        progress_bar(i, max_topics)

    return nmf_models, costs


def main():

    table_name = "govwin_opportunity"

    doc_ids, corpus = get_db_corpus("govwin_opportunity", "opportunity_id", "program_description", remove_html=True)
    count_vectorizer, doc_ids, count_vectorizer_corpus = count_vectorize(doc_ids, corpus, table_name)
    # count_vectorizer, doc_ids, count_vectorizer_corpus = count_vectorize_cache(table_name)

    tfidf_transformer, tfidf_corpus = cv_to_tfidf(doc_ids, count_vectorizer_corpus, table_name)

    nmf_models, costs = search_models(tfidf_corpus, 1, 50)
    print(costs)


if __name__ == "__main__":
    main()
