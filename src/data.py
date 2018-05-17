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



def cache_corpus(table_name, id_column, text_column):

    debug("Connecting to Postgres database...")

    with open("/home/mark/missionmark_db_creds", "r") as f:
        host = f.readline()[:-1]
        dbname = f.readline()[:-1]
        user = f.readline()[:-1]
        password = f.readline()[:-1]

    conn = psycopg2.connect(host=host, dbname=dbname, user=user, password=password)
    debug(" -> Connection successful!", 1)


    corpus_cached_ids = get_cached_corpus_doc_ids(table_name)
    n_cached = len(corpus_cached_ids)
    debug(f" -> {n_cached} already cached...", 1)


    debug("Loading corpus...")
    with conn.cursor() as cursor:

        q = f"""
               SELECT COUNT(*)
               FROM import.{table_name}
               WHERE {text_column} IS NOT NULL
                 AND {text_column} != ''
            """

        cursor.execute(q)
        n_docs = cursor.fetchone()[0] - n_cached


    debug(f" -> Downloading {n_docs}...", 1)
    completed = 0
    with conn.cursor(name="doc_getter") as cursor:
        cursor.itersize = 1000

        q = f"""
                SELECT {id_column}, {text_column}
                FROM import.{table_name}
                WHERE {text_column} IS NOT NULL
                  AND {text_column} != ''
                  AND {id_column} NOT IN ('{"', '".join(corpus_cached_ids)}')
             """

        cursor.execute(q)
        del q  # so pycharm doesn't crash

        for id, doc in cursor:

            if not os.path.exists(f"data/{table_name}/docs/{id}.txt"):
                with open(f"data/{table_name}/docs/{id}.txt", "w") as f:
                    f.write(doc)
                    progress_bar(completed, n_docs, 1)
                completed += 1


        debug(f" -> {n_docs} documents cached!", 1)



def get_corpus():

    if CORPUS_PICKLING:
        debug("Loading cached corpus...")
        with open("pickle/corpus.pkl", "rb") as f:
            corpus = pickle.load(f)

        with open("pickle/ids.pkl", "rb") as f:
            doc_ids = pickle.load(f)

    else:
        debug("Loading corpus...")
        with open("/home/mark/missionmark_db_creds", "r") as f:
            host = f.readline()[:-1]
            dbname = f.readline()[:-1]
            user = f.readline()[:-1]
            password = f.readline()[:-1]

        debug("Connecting to Postgres database...")
        conn = psycopg2.connect(host=host, dbname=dbname, user=user, password=password)
        debug(" -> Connection successful!", 1)


        q = """
               SELECT id, text
               FROM import.fbo_files
               WHERE text IS NOT NULL
                 AND text != ''
               LIMIT 10000
            """

        cursor = conn.cursor()
        cursor.execute(q)

        doc_ids = []
        corpus = []

        for row in cursor:
            if row[0] and row[1]:
                doc_ids.append(row[0])
                corpus.append(row[1])


    debug(f" -> {len(corpus)} documents loaded!", 1)


    if not CORPUS_PICKLING:
        debug("Caching corpus...")
        with open("pickle/doc_ids.pkl", "wb") as f:
            pickle.dump(doc_ids, f)
        with open("pickle/corpus.pkl", "wb") as f:
            pickle.dump(corpus, f)
        debug(" -> Corpus cached!", 2)


    return doc_ids, corpus



def get_cached_corpus_filenames(table_name):
    debug("Searching for cached documents...")
    cached_filenames = [f"data/{table_name}/docs/" + file for file in os.listdir(f"data/{table_name}/docs/") if file.endswith(".txt")]
    n_docs = len(cached_filenames)
    debug(f" -> {n_docs} cached documents found!", 1)

    return cached_filenames, n_docs



def get_cached_corpus_doc_ids(table_name):
    return [file[:-4] for file in os.listdir(f"data/{table_name}/docs/") if file.endswith(".txt")]




if __name__ == "__main__":

    cache_corpus("govwin_opportunity")
