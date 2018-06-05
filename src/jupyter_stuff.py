import numpy as np

from summaries import summarize_doc, summarize_doc_nmf
from IPython.display import Image, display

from nlp import get_top_10_docs
import time

from topics import search_models, plot_results


def top7(topic_i, _W, _corpus, _doc_ids, _tfidf, _nmf, show_wordclouds=True, n_sentences=2):
    if show_wordclouds:
        display(Image(filename=f"../static/wordclouds/{topic_i}_nmf_wordcloud.png"))
        display(Image(filename=f"../static/wordclouds/{topic_i}_tfidf_wordcloud.png"))

    print("TOP 7 MOST RELEVANT DOCUMENTS (auto-summarized)\n")

    i = 0
    for doc, strength in get_top_10_docs(_W, topic_i):
        if i == 7:
            break
        print("*************************************************************")
        print(_doc_ids[doc], " -> STRENGTH: ", round(strength * 100, 2), "%", sep="")
        print(summarize_doc_nmf(_corpus[doc], _tfidf, _nmf, topic_i, n_sentences=n_sentences))
        #print(summarize_doc(doc))
        print()
        i += 1


def term_search(term, _H, _tfidf):
    term_i = _tfidf.vocabulary_[term]
    n_term_topics = np.count_nonzero(_H[:, term_i])
    term_topics = np.argsort(_H[:, term_i])[::-1][:n_term_topics]
    term_topics_strength = _H[:, term_i].flatten()[term_topics]
    for topic_i, strength in zip(term_topics, term_topics_strength):
        print(str(topic_i).rjust(2), " = ", round(strength * 100, 2), "%", sep="")


def topic_search(min_topics, max_topics, tfidf_corpus):

    time_start = time.time()
    costs, H_similarities, W_similarities, tfidf_similarity, max_strength, min_strength, avg_strength = search_models(tfidf_corpus, min_topics, max_topics)
    time_dif = datetime.timedelta(seconds=round(time.time() - time_start))
    plot_results(min_topics, costs, H_similarities, W_similarities, tfidf_similarity, max_strength, min_strength, avg_strength, time_dif, tfidf_corpus.shape)
