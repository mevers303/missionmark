# Mark Evers
# 5/3/18
# acronym_builder.py
# Script for building a dictionary of acronyms.


import psycopg2
import re


conn = psycopg2.connect(host="analysis.c7znlzygfo4d.us-west-2.rds.amazonaws.com", dbname="postgres", user="import_read_user", password="marcoon")
cursor = conn.cursor()


acronyms = {}


q = """select text
from import.fbo_files
where ts_text_simple @@ to_tsquery('agile <-> software <-> development')
limit 10"""

cursor.execute(q)

for row in cursor:

    tokens = row[0].split()

    for i, token in enumerate(tokens):

        matchObj = re.match(r"\(([A-Z0-9\-]+)\)", token)
        if matchObj:
            acronym = matchObj.group(1)
            letters = len(acronym)
            definition = tokens[i - letters:i]

            acronyms[acronym] = definition

        else:
            matchObj = re.match(r"\b\((.*)\)\b", token)
            if matchObj:
                acronym = matchObj.group(1)
                letters = len(acronym)
                definition = tokens[i - letters - 1:i - 1]
                pass
