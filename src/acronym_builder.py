# Mark Evers
# 5/3/18
# acronym_builder.py
# Script for building a dictionary of acronyms.


import psycopg2
from src.acronyms import acronyms_from_doc, add_multiple_to_acronyms, print_acronyms


conn = psycopg2.connect(host="analysis.c7znlzygfo4d.us-west-2.rds.amazonaws.com", dbname="postgres", user="import_read_user", password="marcoon")
cursor = conn.cursor()

q = """select text
from import.fbo_files
where ts_text_simple @@ to_tsquery('agile <-> software <-> development')
limit 10"""


cursor.execute(q)
acronyms = {}

for row in cursor:
    add_multiple_to_acronyms(acronyms, acronyms_from_doc(row[0]))


print_acronyms(acronyms)