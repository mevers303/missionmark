# Mark Evers
# 5/9/2018
# globals.py
# Global variables and functions

from sys import stdout
from time import time


DEBUG_LEVEL = 1
VECTORIZER_MODEL_PICKLING = True
TOPIC_MODEL_PICKLING = True
CORPUS_PICKLING = True
DOC_BUFFER_SIZE = 100000
MAX_FEATURES = 50000
MIN_DF = 1
MAX_DF = .66
N_GRAMS = 1

TABLE_NAME = "govwin_opportunity"
ID_COLUMN = "opportunity_id"
TEXT_COLUMN = "program_description"
TEXT_COLUMN_MIN_LENGTH = 100
STRIP_HTML = True


_PROGRESS_BAR_LAST_TIME = 0
def progress_bar(done, total, resolution=0.333, text=""):
    """
    Prints a progress bar to stdout.
    :param done: Number of items complete
    :param total: Total number if items
    :param resolution: How often to update the progress bar (in seconds).
    :return: None
    """

    global _PROGRESS_BAR_LAST_TIME

    time_now = time()
    if time_now - _PROGRESS_BAR_LAST_TIME < resolution and done < total:
        return

    # percentage done
    i = int(done / total * 100)

    stdout.write('\r')
    # print the progress bar
    stdout.write("[{}]{}%".format(("-" * int(i / 2) + (">" if i < 100 else "")).ljust(50), str(i).rjust(4)))
    # print the text figures
    stdout.write(" ({}/{})".format(done, total).rjust(15))
    if text:
        stdout.write(" " + text)
    stdout.flush()

    if i == 100:
        # print("\n")
        stdout.write('\r')
        stdout.write(' ' * 120)
        stdout.write('\r')

    _PROGRESS_BAR_LAST_TIME = time_now


def debug(text, level = 0):

    if level <= DEBUG_LEVEL:
        print(text)


def get_command_line_options():

    global DEBUG_LEVEL, VECTORIZER_MODEL_PICKLING, TOPIC_MODEL_PICKLING, CORPUS_PICKLING, DOC_BUFFER_SIZE, MAX_FEATURES, MIN_DF, MAX_DF, N_GRAMS, TEXT_COLUMN_MIN_LENGTH, TABLE_NAME, ID_COLUMN, TEXT_COLUMN, STRIP_HTML

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug-level", type=int, help="Numeric value for debug output level.\n-1 is silent\n0 is status updates (default)\nhigher numbers up to 3 will increase the output (default = 1)")
    parser.add_argument("--table-name", help="The table to use for database queries (default = \"govwin_opportunity\")")
    parser.add_argument("--id-column", help="The column to use for the document IDs in database queries (default = \"opportunity_id\")")
    parser.add_argument("--text-column", help="The column to use for the document text in database queries (default = \"program_description\")")
    parser.add_argument("--min-doc-length", type=int, help="The minimum character length of the text column in database queries (default = 100)")
    parser.add_argument("--strip-html", action="store_true", help="Whether or not to strip HTML from the text column (default = False)")
    parser.add_argument("--vectorizer-pickling", type=int, choices=[0, 1], help="Whether or not to load cached vectorizer models (default = True)")
    parser.add_argument("--topic-pickling", type=int, choices=[0, 1], help="Whether or not to load cached topic models (default = True)")
    parser.add_argument("--corpus-pickling", type=int, choices=[0, 1], help="Whether or not to load cached corpora (default = True)")
    parser.add_argument("--cursor-size", type=int, help="The size of the database cursor buffer (default = 100000)")
    parser.add_argument("--max-features", type=int, help="The maximum number of features (keywords) to use in the vectorizers (default = 50000)")
    parser.add_argument("--min-df", type=int, help="The minimum document frequency for a keyword in the vectorizers (default = 1)")
    parser.add_argument("--max-df", type=float, help="The maximum document frequency for a keyword in the vectorizers (default = 0.66)")
    parser.add_argument("--n-grams", type=int, help="The size of the N-grams to extract from the corpora (default = 1)")

    args = parser.parse_args()
    if args.debug_level:
        DEBUG_LEVEL = args.debug_level
    if args.table_name:
        TABLE_NAME = args.table_name
    if args.id_column:
        ID_COLUMN = args.id_column
    if args.text_column:
        TEXT_COLUMN = args.text_column
    if args.strip_html:
        STRIP_HTML = args.strip_html
    if args.vectorizer_pickling:
        VECTORIZER_MODEL_PICKLING = args.vectorizer_pickling
    if args.topic_pickling:
        TOPIC_MODEL_PICKLING = args.topic_pickling
    if args.corpus_pickling:
        CORPUS_PICKLING = args.corpus_pickling
    if args.cursor_size:
        DOC_BUFFER_SIZE = args.cursor_size
    if args.max_features:
        MAX_FEATURES = args.max_features
    if args.min_df:
        MIN_DF = args.min_df
    if args.max_df:
        MAX_DF = args.max_df
    if args.n_grams:
        N_GRAMS = args.n_grams

    return args


args = get_command_line_options()
