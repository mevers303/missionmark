# Mark Evers
#  5/20/18


import numpy as np

from nlp import nmf_model, get_corpus_top_topics
import globals as g
from vectorizer import get_cached_corpus
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
import time
import datetime




def search_models(tfidf_corpus, min_topics, max_topics, table_name):

    g.debug("Building NMF topics...")
    # nmf_models = []
    costs = []
    intertopic_similarities = []
    interdocument_similarities = []
    n_models = max_topics - min_topics + 1

    g.progress_bar(0, n_models)
    for i in range(min_topics, max_topics + 1):

        nmf, W, H = nmf_model(tfidf_corpus, i, table_name, False, no_output=True)
        top_topics = get_corpus_top_topics(W)

        # nmf_models.append(nmf)
        costs.append(nmf.reconstruction_err_**2)
        intertopic_similarities.append(1 - pairwise_distances(H, metric="cosine", n_jobs=-1).mean())
        interdocument_similarities.append(np.mean([1 - pairwise_distances(tfidf_corpus[top_topics == topic_i].A, metric="cosine", n_jobs=-1).mean() for topic_i in range(i) if np.isin(topic_i, top_topics)]))

        g.progress_bar(i - min_topics + 1, n_models, text=f"{nmf.n_iter_} iterations")


    return costs, intertopic_similarities, interdocument_similarities


def plot_results(min_topics, max_topics, costs, intertopic_similarities, interdocument_similarities, time, shape):

    fig, axes = plt.subplots(3, 1, figsize=(30, 20))
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

    plt.suptitle(f"n_topics ({shape[0]} docs, {shape[1]} features, {g.N_GRAMS} n-grams) [{time}]")
    # plt.tight_layout()
    plt.savefig(f"../output/{g.TABLE_NAME}/nmf/n_topics_nmf.png")
    # plt.show()


def main():

    g.get_command_line_options()
    print(g.N_GRAMS)

    min_topics = 5
    max_topics = 160

    doc_ids, tfidf_corpus = get_cached_corpus(g.TABLE_NAME, "tfidf")
    time_start = time.time()
    costs, intertopic_similarities, interdocument_similarities = search_models(tfidf_corpus, min_topics, max_topics, g.TABLE_NAME)
    time_dif = datetime.timedelta(seconds=round(time.time() - time_start))
    plot_results(min_topics, max_topics, costs, intertopic_similarities, interdocument_similarities, time_dif, tfidf_corpus.shape)


if __name__ == "__main__":
    main()
