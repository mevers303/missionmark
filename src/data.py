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

    q = """
           select text
           from import.fbo_files
           where ts_text_simple @@ to_tsquery('agile <-> software <-> development')
        """

    cursor.execute(q)
    return [row[0] for row in cursor]



def get_data():

    print("Loading data...")

    q = """
           select name, description, text
           from import.fbo_files
           where ts_text_simple @@ to_tsquery('agile <-> software <-> development')
           limit 100
        """

    cursor.execute(q)

    names = []
    descriptions = []
    docs = []

    for row in cursor:
        names.append(row[0])
        descriptions.append(row[1])
        docs.append(row[2])

    return names, descriptions, docs
