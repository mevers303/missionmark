# Mark Evers
# 5/3/18
# acronym_builder.py
# Script for building a dictionary of acronyms.


from src.data import get_test_data
from src.acronyms import acronyms_from_doc, add_multiple_to_acronyms, print_acronyms


acronyms = {}

corpus = get_test_data()
for doc in corpus:
    add_multiple_to_acronyms(acronyms, acronyms_from_doc(doc))


print_acronyms(acronyms)
