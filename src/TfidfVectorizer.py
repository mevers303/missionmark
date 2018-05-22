# Mark Evers
# 5/21/18

class TfidfVectorizer():

    def __init__(self, count_vectorizer, tfidf_transformer):
        self._count_vectorizer = count_vectorizer
        self._tfidf_transformer = tfidf_transformer


    def transform(self, corpus):
        cv_corpus = self._count_vectorizer.transform(corpus)
        return self._tfidf_transformer.transform(cv_corpus)
