# Mark Evers
# 5/3/18
# acronym_builder.py
# Script for building a dictionary of acronyms.


from data import *
from acronyms import acronyms_from_doc, add_multiple_to_acronyms, print_acronyms
from globals import *


acronyms = {}

doc_ids, corpus = get_corpus()
completed = 0
total = len(corpus)

debug("Parsing acronyms...")
for doc in corpus:
    add_multiple_to_acronyms(acronyms, acronyms_from_doc(doc))
    completed += 1
    progress_bar(completed, total, 1)
debug(f" -> {len(acronyms.keys())} acronyms parsed!", 1)


print_acronyms(acronyms)
