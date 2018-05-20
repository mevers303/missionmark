# Mark Evers
# 5/3/18
# data.py
# Script for getting data


import os
import sys
sys.path.append(os.getcwd())
sys.path.append("src")


import psycopg2
from src.globals import *
import pickle

from bs4 import BeautifulSoup
from bs4.element import Comment



_connection = None

def get_connection():

    global _connection

    if not _connection:
        debug("Connecting to Postgres database...")

        with open("missionmark_db_creds", "r") as f:
            host = f.readline()[:-1]
            dbname = f.readline()[:-1]
            user = f.readline()[:-1]
            password = f.readline()[:-1]

        _connection = psycopg2.connect(host=host, dbname=dbname, user=user, password=password)
        debug(" -> Connection successful!", 1)

    return _connection





def cache_corpus(table_name, id_column, text_column, remove_html=False):

    conn = get_connection()
    debug("Loading corpus...")

    corpus_cached_ids = get_cached_doc_ids(table_name)
    n_cached = len(corpus_cached_ids)
    debug(f" -> {n_cached} already cached...", 1)


    with conn.cursor() as cursor:

        q = f"""
               SELECT COUNT(*)
               FROM import.{table_name}
               WHERE LENGTH({text_column}) > {TEXT_COLUMN_MIN_LENGTH}
                 AND {id_column} IS NOT NULL
            """

        cursor.execute(q)
        n_docs = cursor.fetchone()[0]


    debug(f" -> Downloading {n_docs}...", 1)
    with conn.cursor(name="doc_getter") as cursor:
        cursor.itersize = DOC_BUFFER_SIZE

        q = f"""
                SELECT {id_column}, {text_column}
                FROM import.{table_name}
                WHERE LENGTH({text_column}) > {TEXT_COLUMN_MIN_LENGTH}
                  AND {id_column} IS NOT NULL
             """

        cursor.execute(q)

        completed = 0
        for doc_id, doc in cursor:

            if not os.path.exists(f"data/{table_name}/docs/{doc_id}.txt"):
                with open(f"data/{table_name}/docs/{doc_id}.txt", "w") as f:
                    if remove_html:
                        doc = strip_html(doc)
                    f.write(doc)

            completed += 1
            progress_bar(completed, n_docs, 1)


        debug(f" -> {n_docs} documents cached!", 1)



def get_db_corpus(table_name, id_column, text_column, remove_html=False):

    conn = get_connection()
    debug("Loading corpus...")

    with conn.cursor() as cursor:

        q = f"""
               SELECT COUNT(*)
               FROM import.{table_name}
               WHERE LENGTH({text_column}) > {TEXT_COLUMN_MIN_LENGTH}
                 AND {id_column} IS NOT NULL
            """

        cursor.execute(q)
        n_docs = cursor.fetchone()[0]


    debug(f" -> Downloading {n_docs}...", 1)
    with conn.cursor(name="doc_getter") as cursor:
        cursor.itersize = DOC_BUFFER_SIZE

        q = f"""
                SELECT {id_column}, {text_column}
                FROM import.{table_name}
                WHERE LENGTH({text_column}) > {TEXT_COLUMN_MIN_LENGTH}
                  AND {id_column} IS NOT NULL
             """

        cursor.execute(q)
        doc_ids = []
        corpus = []

        for doc_id, doc in cursor:
            if remove_html:
                doc = strip_html(doc)
            doc_ids.append(doc_id)
            corpus.append(doc)

    debug(f" -> {len(corpus)} documents loaded!", 1)

    return doc_ids, corpus



def get_cached_filenames(table_name):
    debug("Searching for cached documents...")
    cached_filenames = [f"data/{table_name}/docs/" + file for file in os.listdir(f"data/{table_name}/docs/") if file.endswith(".txt")]
    n_docs = len(cached_filenames)
    debug(f" -> {n_docs} cached documents found!", 1)

    return cached_filenames, n_docs



def get_cached_doc_ids(table_name):
    return [file[:-4] for file in os.listdir(f"data/{table_name}/docs/") if file.endswith(".txt")]




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
