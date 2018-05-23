# Mark Evers
# 5/3/18
# data.py
# Script for getting data

import os
import psycopg2
import pandas as pd
from bs4 import BeautifulSoup
from bs4.element import Comment

import globals as g




_connection = None

def get_connection():

    global _connection

    if not _connection:
        g.debug("Connecting to Postgres database...")

        with open("../missionmark_db_creds", "r") as f:
            host = f.readline()[:-1]
            dbname = f.readline()[:-1]
            user = f.readline()[:-1]
            password = f.readline()[:-1]

        _connection = psycopg2.connect(host=host, dbname=dbname, user=user, password=password)
        g.debug(" -> Connection successful!", 1)

    return _connection



def get_n_docs(table_name, id_column, text_column):

    conn = get_connection()
    g.debug("Loading corpus...")

    with conn.cursor() as cursor:

        q = f"""
               SELECT COUNT(*)
               FROM import.{table_name}
               WHERE LENGTH({text_column}) > {g.TEXT_COLUMN_MIN_LENGTH}
                 AND {id_column} IS NOT NULL
            """

        cursor.execute(q)
        n_docs = cursor.fetchone()[0]


    g.debug(f" -> Found {n_docs}!", 1)
    return n_docs



def cache_corpus(table_name, id_column, text_column, remove_html=False):

    conn = get_connection()
    n_docs = get_n_docs(table_name, id_column, text_column)


    corpus_cached_ids = get_cached_doc_ids(table_name)
    n_cached = len(corpus_cached_ids)
    g.debug(f" -> {n_cached} already cached...", 1)


    with conn.cursor(name="doc_getter") as cursor:
        cursor.itersize = g.DOC_BUFFER_SIZE

        q = f"""
                SELECT {id_column}, {text_column}
                FROM import.{table_name}
                WHERE LENGTH({text_column}) > {g.TEXT_COLUMN_MIN_LENGTH}
                  AND {id_column} IS NOT NULL
             """

        cursor.execute(q)

        completed = 0
        for doc_id, doc in cursor:

            if not os.path.exists(f"../data/{table_name}/docs/{doc_id}.txt"):
                with open(f"../data/{table_name}/docs/{doc_id}.txt", "w") as f:
                    if remove_html:
                        doc = strip_html(doc)
                    f.write(doc)

            completed += 1
            g.progress_bar(completed, n_docs, 1)


        g.debug(f" -> {n_docs} documents cached!", 1)



def get_db_corpus(table_name, id_column, text_column, remove_html=False):

    conn = get_connection()
    n_docs = get_n_docs(table_name, id_column, text_column)


    with conn.cursor(name="doc_getter") as cursor:
        cursor.itersize = g.DOC_BUFFER_SIZE

        q = f"""
                SELECT {id_column}, {text_column}
                FROM import.{table_name}
                WHERE LENGTH({text_column}) > {g.TEXT_COLUMN_MIN_LENGTH}
                  AND {id_column} IS NOT NULL
             """

        cursor.execute(q)

        corpus_df = pd.DataFrame(columns=["text"])
        completed = 0

        for doc_id, doc in cursor:
            corpus_df.loc[doc_id] = doc if remove_html else strip_html(doc)
            completed += 1
            g.progress_bar(completed, n_docs, 1)


        g.debug(f" -> {n_docs} documents loaded!", 1)


    return corpus_df



def get_cached_filenames(table_name):

    g.debug("Searching for cached documents...")
    corpus_df = pd.DataFrame(columns=["file"])

    for file in os.listdir(f"../data/{table_name}/docs/"):
        if not file.endswith(".txt"):
            continue
        corpus_df.loc[file[:-4]] = f"../data/{table_name}/docs/{file}"


    g.debug(f" -> {corpus_df.shape[0]} cached documents found!", 1)

    return corpus_df



def get_cached_doc_ids(table_name):
    return [file[:-4] for file in os.listdir(f"../data/{table_name}/docs/") if file.endswith(".txt")]



def dump_doc_ids(doc_ids, filename):

    with open(filename, "w") as f:
        for doc_id in doc_ids:
            f.write(str(doc_id) + "\n")


def load_doc_ids(filename):
    return [line[:-1] for line in open(filename, "r")]


def check_corpus_pickles(table_name):

    return g.CORPUS_PICKLING and \
           os.path.exists(f"../data/{table_name}/pickles/TfidfTransformer.pkl") and \
           os.path.exists(f"../data/{table_name}/pickles/TfidfTransformer_corpus.pkl") and \
           os.path.exists(f"../data/{table_name}/pickles/TfidfTransformer_doc_ids.txt")



def _tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def strip_html(doc):
    soup = BeautifulSoup(doc, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(_tag_visible, texts)
    return " ".join(t.strip() for t in visible_texts)





if __name__ == "__main__":

    cache_corpus("govwin_opportunity", "opportunity_id", "program_description", remove_html=True)
    # cache_corpus("fbo_files", "id", "text", remove_html=False)
