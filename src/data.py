# Mark Evers
# 5/3/18
# data.py
# Script for getting data


import psycopg2
from src.globals import *
import pickle



def get_corpus():


    with open("/home/mark/missionmark_db_creds", "r") as f:
        host = f.readline()[:-1]
        dbname = f.readline()[:-1]
        user = f.readline()[:-1]
        password = f.readline()[:-1]

    debug("Connecting to Postgres database...")
    conn = psycopg2.connect(host=host, dbname=dbname, user=user, password=password)
    debug(" -> Connection successful!", 1)

    debug("Loading corpus...")

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
    return doc_ids, corpus



def load_pickle_corpus():

    with open("pickle/corpus.pkl", "rb") as f:
        corpus = pickle.load(f)

    with open("pickle/ids.pkl", "rb") as f:
        doc_ids = pickle.load(f)

    return doc_ids, corpus



if __name__ == "__main__":

    ids, corpus = get_corpus()
