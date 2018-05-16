# Mark Evers
# 5/3/18
# data.py
# Script for getting data


import psycopg2
from src.globals import *
import pickle
import os



def cache_corpus():

    debug("Connecting to Postgres database...")

    with open("/home/mark/missionmark_db_creds", "r") as f:
        host = f.readline()[:-1]
        dbname = f.readline()[:-1]
        user = f.readline()[:-1]
        password = f.readline()[:-1]

    conn = psycopg2.connect(host=host, dbname=dbname, user=user, password=password)
    debug(" -> Connection successful!", 1)


    debug("Loading corpus...")
    cursor = conn.cursor()

    q = """
           SELECT COUNT(*)
           FROM import.fbo_files
        """

    cursor.execute(q)
    n_docs = cursor.fetchone()[0]
    debug(f"Found {n_docs}!")


    completed = 0
    for i in range(0, n_docs, DOC_BUFFER_SIZE):

        q = f"""
                SELECT id, text
                FROM import.fbo_files
                WHERE text IS NOT NULL
                  AND text != ''
                LIMIT {DOC_BUFFER_SIZE} OFFSET {i}
             """

        cursor.execute(q)

        for id, doc in cursor:

            with open(os.path.join("data/docs", f"{id}.txt"), "w") as f:
                f.write(doc)
                completed += 1
                progress_bar(completed, n_docs, 1)


    debug(f" -> {len(corpus)} documents loaded!", 1)



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



if __name__ == "__main__":

    cache_corpus()
