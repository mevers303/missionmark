# Mark Evers
# 5/3/18
# acronym_builder.py
# Script for building a dictionary of acronyms.


from data import get_db_corpus
from acronyms import acronyms_from_doc, add_multiple_to_acronyms, print_acronyms
from globals import *


acronyms = {}

corpus_df = get_db_corpus("govwin_opportunity", "opportunity_id", "program_description", remove_html=True)
completed = 0
total = corpus_df.shape[0]

debug("Parsing acronyms...")
for doc in corpus_df.values[:, 0]:
    add_multiple_to_acronyms(acronyms, acronyms_from_doc(doc))
    completed += 1
    progress_bar(completed, total, 1)
debug(f" -> {len(acronyms.keys())} acronyms parsed!", 1)


print_acronyms(acronyms)
