# Mark Evers
#  5/20/18


import numpy as np

from nlp import nmf_model
from globals import *
from vectorizer import count_vectorize_cache, cv_to_tfidf, count_vectorize, get_cached_corpus
from data import get_db_corpus, check_corpus_pickles
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt



def rss_cost(V, W, H):
    return (np.array(V - W.dot(H))**2).sum()


def search_models(tfidf_corpus, min_topics, max_topics):

    debug("Building NMF topics...")
    nmf_models = []
    costs = []
    intertopic_similarities = []
    interdocument_similarities = []
    n_models = max_topics - min_topics

    progress_bar(0, n_models)
    for i in range(min_topics, max_topics + 1):

        nmf, W, H = nmf_model(tfidf_corpus, i, skip_pickling=True)
        nmf_models.append(nmf)
        costs.append(rss_cost(tfidf_corpus, W, H))
        intertopic_similarities.append(pdist(H, "cosine").mean())
        interdocument_similarities.append(pdist(W, "cosine").mean())
        progress_bar(i - min_topics + 1, n_models)

    return nmf_models, costs, intertopic_similarities, interdocument_similarities


def main():

    min_topics = 10
    max_topics = 50

    doc_ids, tfidf_corpus = get_cached_corpus(TABLE_NAME, "tfidf")
    nmf_models, costs, intertopic_similarities, interdocument_similarities = search_models(tfidf_corpus, min_topics, max_topics)

    fig, axes = plt.subplots(3, 1)
    axes = axes.flatten()
    x = np.arange(min_topics, max_topics + 1)

    axes[0].set_title("RSS")
    axes[0].set_xlabel("Topics")
    axes[0].set_ylabel("RSS")
    axes[0].plot(x, costs)

    axes[1].set_title("Intertopic Cosine Similarity")
    axes[1].set_xlabel("Topics")
    axes[1].set_ylabel("Cosine Similarity")
    axes[1].plot(x, intertopic_similarities)

    axes[1].set_title("Interdocument Cosine Similarity")
    axes[1].set_xlabel("Topics")
    axes[1].set_ylabel("Cosine Similarity")
    axes[1].plot(x, interdocument_similarities)




if __name__ == "__main__":
    main()
