# Mark Evers
# 5/21/18

class TfidfVectorizer():

    def __init__(self, count_vectorizer, tfidf_transformer):
        self._count_vectorizer = count_vectorizer
        self._tfidf_transformer = tfidf_transformer
        self.vocabulary_ = self._count_vectorizer.vocabulary_
        self.stop_words_ = self._count_vectorizer.stop_words_


    def transform(self, corpus):
        cv_corpus = self._count_vectorizer.transform(corpus)
        return self._tfidf_transformer.transform(cv_corpus)

    def get_feature_names(self):
        return self._count_vectorizer.get_feature_names()
