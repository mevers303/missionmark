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
cursor = conn.cursor()
debug(" -> Connection successful!", 1)



def get_test_data():

    debug("Loading test corpus...")

    q = """
           select text
           from import.fbo_files
           where ts_text_simple @@ to_tsquery('agile <-> software <-> development')
        """

    cursor.execute(q)
    corpus = [row[0] for row in cursor]

    debug(f" -> {len(corpus)} documents loaded!", 1)
    return corpus



def get_all_data():

    debug("Loading corpus...")

    q = """
           select id, text
           from import.fbo_files
        """

    cursor.execute(q)

    ids, corpus = zip(*cursor)

    # ids = []
    # docs = []
    #
    # for row in cursor:
    #     ids.append(row[0])
    #     docs.append(row[1])

    debug(f" -> {len(corpus)} documents loaded!", 1)

    return ids, corpus


if __name__ == "__main__":

    ids, corpus = get_all_data()
