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




def search_models(tfidf_corpus, min_topics, max_topics, threshold=.33):

    g.debug("Building NMF topics...")
    # nmf_models = []
    costs = []
    intertopic_similarities = []
    interdocument_similarities = []
    tfidf_similarities = []
    n_models = max_topics - min_topics + 1

    g.progress_bar(0, n_models)
    for i in range(min_topics, max_topics + 1):

        try:
            nmf, W, H = nmf_model(tfidf_corpus, i, no_output=True)

            # nmf_models.append(nmf)
            costs.append(nmf.reconstruction_err_**2)
            intertopic_similarities.append(1 - pairwise_distances(H, metric="cosine", n_jobs=-1).mean())
            interdocument_similarities.append(1 - pairwise_distances(W, metric="cosine", n_jobs=-1).mean())
            W_normalized = W / W.max(axis=0)
            tfidf_similarities.append(np.mean([1 - pairwise_distances(tfidf_corpus[W_normalized[:, topic_i] > threshold].A, metric="cosine", n_jobs=-1).mean() for topic_i in range(i) if (W_normalized[:, topic_i] > threshold).any()]))

            g.progress_bar(i - min_topics + 1, n_models, text=f"{nmf.n_iter_} iterations")

        except KeyboardInterrupt:
            completed = len(interdocument_similarities)
            costs = costs[:completed]
            intertopic_similarities = intertopic_similarities[:completed]
            break


    return costs, intertopic_similarities, interdocument_similarities, tfidf_similarities


def plot_results(min_topics, costs, intertopic_similarities, interdocument_similarities, tfidf_similarity, time, shape):

    fig, axes = plt.subplots(4, 1, figsize=(30, 20))
    axes = axes.flatten()
    x = np.arange(min_topics, min_topics + len(costs))

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

    axes[3].set_title("TF/IDF Cosine Similarity")
    axes[3].set_xlabel("Topics")
    axes[3].set_ylabel("Cosine Similarity")
    axes[3].plot(x, tfidf_similarity)

    plt.suptitle(f"n_topics ({shape[0]} docs, {shape[1]} features, {g.N_GRAMS} n-grams) [{time}]")
    # plt.tight_layout()
    plt.savefig(f"../output/topic_search.png")
    return fig


def main():

    g.get_command_line_options()

    min_topics = 5
    max_topics = 160

    doc_ids, tfidf_corpus = get_cached_corpus(g.TABLE_NAME, "tfidf")
    time_start = time.time()
    costs, intertopic_similarities, interdocument_similarities, tfidf_similarity = search_models(tfidf_corpus, min_topics, max_topics)
    time_dif = datetime.timedelta(seconds=round(time.time() - time_start))
    plot_results(min_topics, costs, intertopic_similarities, interdocument_similarities, tfidf_similarity, time_dif, tfidf_corpus.shape)
    plt.show()


if __name__ == "__main__":
    main()
