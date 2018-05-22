# Mark Evers
#  5/20/18


import numpy as np

from nlp import nmf_model, get_corpus_top_topics
from globals import *
from vectorizer import count_vectorize_cache, cv_to_tfidf, count_vectorize, get_cached_corpus
from data import get_db_corpus, check_corpus_pickles
from sklearn.metrics.pairwise import pairwise_distances as pdist
import matplotlib.pyplot as plt




def search_models(tfidf_corpus, min_topics, max_topics):

    debug("Building NMF topics...")
    nmf_models = []
    reconstruction_errors = []
    costs = []
    intertopic_similarities = []
    interdocument_similarities = []
    n_models = max_topics - min_topics + 1

    progress_bar(0, n_models)
    for i in range(min_topics, max_topics + 1):

        nmf, W, H = nmf_model(tfidf_corpus, i, "", False, no_output=True)
        top_topics = get_corpus_top_topics(W)

        nmf_models.append(nmf)
        costs.append(nmf.reconstruction_err_**2)
        intertopic_similarities.append(1 - pdist(H, metric="cosine", n_jobs=-1).mean())
        interdocument_similarities = np.mean([1 - pdist(tfidf_corpus[top_topics == topic_i].A, metric="cosine", n_jobs=-1).mean() for topic_i in range(i)])

        progress_bar(i - min_topics + 1, n_models, f"{nmf.n_iter_} iterations")

    return nmf_models, costs, intertopic_similarities, interdocument_similarities


def main():

    min_topics = 5
    max_topics = 10

    doc_ids, tfidf_corpus = get_cached_corpus(TABLE_NAME, "tfidf")
    nmf_models, costs, intertopic_similarities, interdocument_similarities = search_models(tfidf_corpus, min_topics, max_topics)

    fig, axes = plt.subplots(4, 1)
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

    axes[2].set_title("Interdocument Cosine Similarity")
    axes[2].set_xlabel("Topics")
    axes[2].set_ylabel("Cosine Similarity")
    axes[2].plot(x, interdocument_similarities)

    plt.tight_layout()
    plt.show()
    print("yeah")




if __name__ == "__main__":
    main()
