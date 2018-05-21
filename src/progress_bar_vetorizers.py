# Mark Evers
# 5/18/18

import numpy
from sys import stdout
from sklearn.feature_extraction.text import CountVectorizer
import time



class CountVectorizerProgressBar(CountVectorizer):
    """
        scikit-learn's CountVectorizer object, but it displays a progress bar when fitting/transforming!
            progress_bar_resolution is how long (in seconds) it should wait in between updating the progress bar.
            progress_bar_clear (boolean) is whether or not to clear the progress bar displaying 100% from stdout when it
                is done.

        Author: Mark Evers
        Github: http://github.com/mevers303

        Convert a collection of text documents to a matrix of token counts
        This implementation produces a sparse representation of the counts using
        scipy.sparse.csr_matrix.
        If you do not provide an a-priori dictionary and you do not use an analyzer
        that does some kind of feature selection then the number of features will
        be equal to the vocabulary size found by analyzing the data.
        Read more in the :ref:`User Guide <text_feature_extraction>`.
        Parameters
        ----------
        input : string {'filename', 'file', 'content'}
            If 'filename', the sequence passed as an argument to fit is
            expected to be a list of filenames that need reading to fetch
            the raw content to analyze.
            If 'file', the sequence items must have a 'read' method (file-like
            object) that is called to fetch the bytes in memory.
            Otherwise the input is expected to be the sequence strings or
            bytes items are expected to be analyzed directly.
        encoding : string, 'utf-8' by default.
            If bytes or files are given to analyze, this encoding is used to
            decode.
        decode_error : {'strict', 'ignore', 'replace'}
            Instruction on what to do if a byte sequence is given to analyze that
            contains characters not of the given `encoding`. By default, it is
            'strict', meaning that a UnicodeDecodeError will be raised. Other
            values are 'ignore' and 'replace'.
        strip_accents : {'ascii', 'unicode', None}
            Remove accents during the preprocessing step.
            'ascii' is a fast method that only works on characters that have
            an direct ASCII mapping.
            'unicode' is a slightly slower method that works on any characters.
            None (default) does nothing.
        analyzer : string, {'word', 'char', 'char_wb'} or callable
            Whether the feature should be made of word or character n-grams.
            Option 'char_wb' creates character n-grams only from text inside
            word boundaries; n-grams at the edges of words are padded with space.
            If a callable is passed it is used to extract the sequence of features
            out of the raw, unprocessed input.
        preprocessor : callable or None (default)
            Override the preprocessing (string transformation) stage while
            preserving the tokenizing and n-grams generation steps.
        tokenizer : callable or None (default)
            Override the string tokenization step while preserving the
            preprocessing and n-grams generation steps.
            Only applies if ``analyzer == 'word'``.
        ngram_range : tuple (min_n, max_n)
            The lower and upper boundary of the range of n-values for different
            n-grams to be extracted. All values of n such that min_n <= n <= max_n
            will be used.
        stop_words : string {'english'}, list, or None (default)
            If 'english', a built-in stop word list for English is used.
            If a list, that list is assumed to contain stop words, all of which
            will be removed from the resulting tokens.
            Only applies if ``analyzer == 'word'``.
            If None, no stop words will be used. max_df can be set to a value
            in the range [0.7, 1.0) to automatically detect and filter stop
            words based on intra corpus document frequency of terms.
        lowercase : boolean, True by default
            Convert all characters to lowercase before tokenizing.
        token_pattern : string
            Regular expression denoting what constitutes a "token", only used
            if ``analyzer == 'word'``. The default regexp select tokens of 2
            or more alphanumeric characters (punctuation is completely ignored
            and always treated as a token separator).
        max_df : float in range [0.0, 1.0] or int, default=1.0
            When building the vocabulary ignore terms that have a document
            frequency strictly higher than the given threshold (corpus-specific
            stop words).
            If float, the parameter represents a proportion of documents, integer
            absolute counts.
            This parameter is ignored if vocabulary is not None.
        min_df : float in range [0.0, 1.0] or int, default=1
            When building the vocabulary ignore terms that have a document
            frequency strictly lower than the given threshold. This value is also
            called cut-off in the literature.
            If float, the parameter represents a proportion of documents, integer
            absolute counts.
            This parameter is ignored if vocabulary is not None.
        max_features : int or None, default=None
            If not None, build a vocabulary that only consider the top
            max_features ordered by term frequency across the corpus.
            This parameter is ignored if vocabulary is not None.
        vocabulary : Mapping or iterable, optional
            Either a Mapping (e.g., a dict) where keys are terms and values are
            indices in the feature matrix, or an iterable over terms. If not
            given, a vocabulary is determined from the input documents. Indices
            in the mapping should not be repeated and should not have any gap
            between 0 and the largest index.
        binary : boolean, default=False
            If True, all non zero counts are set to 1. This is useful for discrete
            probabilistic models that model binary events rather than integer
            counts.
        dtype : type, optional
            Type of the matrix returned by fit_transform() or transform().
        Attributes
        ----------
        vocabulary_ : dict
            A mapping of terms to feature indices.
        stop_words_ : set
            Terms that were ignored because they either:
              - occurred in too many documents (`max_df`)
              - occurred in too few documents (`min_df`)
              - were cut off by feature selection (`max_features`).
            This is only available if no vocabulary was given.
        See also
        --------
        HashingVectorizer, TfidfVectorizer
        Notes
        -----
        The ``stop_words_`` attribute can get large and increase the model size
        when pickling. This attribute is provided only for introspection and can
        be safely removed using delattr or set to None before pickling.
        """


    def __init__(self, input="content", encoding="utf-8", decode_error="strict", strip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, stop_words=None, token_pattern="(?u)\b\w\w+\b", ngram_range=(1, 1),
                 analyzer="word", max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False,
                 dtype=numpy.int64, progress_bar_resolution=.333, progress_bar_clear=False):

        super().__init__(input=input, encoding=encoding, decode_error=decode_error, strip_accents=strip_accents,
                         lowercase=lowercase, preprocessor=preprocessor, tokenizer=tokenizer, stop_words=stop_words,
                         token_pattern=token_pattern, ngram_range=ngram_range, analyzer=analyzer, max_df=max_df,
                         min_df=min_df, max_features=max_features, vocabulary=vocabulary, binary=binary, dtype=dtype)
        self._n_docs = 0
        self._completed_docs = 0
        self._progress_bar_last_time = 0
        self._progress_bar_resolution = progress_bar_resolution
        self._progress_bar_clear = progress_bar_clear


    def _analyzer_with_progress_bar(self, analyzer, doc):
        result = analyzer(doc)
        self._completed_docs += 1
        self.progress_bar()
        return result


    def build_analyzer(self):
        analyzer = super().build_analyzer()
        return lambda doc: self._analyzer_with_progress_bar(analyzer, doc)


    def fit(self, raw_documents, y=None):
        self._n_docs = len(raw_documents)
        self._completed_docs = 0
        self.progress_bar()
        return super().fit(raw_documents, y=y)


    def fit_transform(self, raw_documents, y=None):
        self._n_docs = len(raw_documents)
        self._completed_docs = 0
        self.progress_bar()
        return super().fit_transform(raw_documents, y=y)


    def transform(self, raw_documents):
        self._n_docs = len(raw_documents)
        self._completed_docs = 0
        self.progress_bar()
        return super().transform(raw_documents)


    def progress_bar(self):

        time_now = time.time()
        if time_now - self._progress_bar_last_time < self._progress_bar_resolution and self._completed_docs < self._n_docs:
            return

        # percentage done
        percentage = int(self._completed_docs / self._n_docs * 100)

        stdout.write('\r')
        # print the progress bar
        stdout.write("[{}]{}%".format(("-" * int(percentage / 2) + (">" if percentage < 100 else "")).ljust(50), str(percentage).rjust(4)))
        # print the text figures
        stdout.write(" ({}/{})".format(self._completed_docs, self._n_docs).rjust(14))
        stdout.flush()

        if percentage == 100 and self._progress_bar_clear:
            # print("\n")
            stdout.write('\r')
            stdout.write(' ' * 80)
            stdout.write('\r')

        self._progress_bar_last_time = time_now
