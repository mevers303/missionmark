# Mark Evers
# 5/3/18
# data.py
# Script for getting data


import psycopg2
from src.globals import *



with open("/home/mark/missionmark_db_creds", "r") as f:
    host = f.readline()[:-1]
    dbname = f.readline()[:-1]
    user = f.readline()[:-1]
    password = f.readline()[:-1]

debug("Connecting to Postgres database...")
conn = psycopg2.connect(host=host, dbname=dbname, user=user, password=password)
debug(" -> Connection successful!", 1)



def get_corpus():

    debug("Loading test corpus...")

    q = """
           SELECT id, text
           FROM import.fbo_files
           LIMIT 10000
        """

    cursor = conn.cursor()
    cursor.execute(q)

    doc_ids = []
    corpus = []

    for row in cursor:
        doc_ids.append(row[0])
        doc_ids.append(row[1])

    debug(f" -> {len(corpus)} documents loaded!", 1)
    return list(doc_ids), list(corpus)


if __name__ == "__main__":

    ids, corpus = get_corpus()
