# Mark Evers
# 5/3/18
# acronyms.py
# Script for building a dictionary of acronyms.


import re
import string
from src.nlp import split_sentences, split_tokens_soft

# These constants are for when it has to search for the acronym
UNMATCHED_ACRONYM_FIRSTLETTER_SEARCH_LEN = 5  # how many words before the believed first word to search
UNMATCHED_ACRONYM_CAPITALS_SEARCH_LEN = 7  # how many capitalized letters to include when above doesn't work
MAX_ACRONYM_LEN = 7  # any longer than this will be ignored



def add_to_acronyms(acronyms, acronym, definition):
    """
    Adds an acronym and it's definition to a dictionary of acronyms.  If there is an overlap, the newer acronym is
    numbered (ie "USA" -> "USA (2)").
    :param acronyms: The dictionary of acronyms.  Passed by reference by default, so it will be modified in place.
    :param acronym: The acronym to be added.
    :param definition: The definition of the acronym to be added.
    :return: The acronym dictionary, but it is not necessary to reassign your local variable because it is modified in
             place.
    """

    key = acronym
    x = 2

    while key in acronyms:

        if acronyms[key].lower() == definition.lower():
            break

        key = acronym + " ({0})".format(x)
        x += 1

    acronyms[key] = definition
    return acronyms



def add_multiple_to_acronyms(acronyms, new_acronyms):
    """
    Merges two acronym dictionaries.  Overlaps are renamed.
    :param acronyms: The acronyms will be added to this dictionary.
    :param new_acronyms: The dictionary of acronyms to be added.
    :return: The acronym dictionary, but it is not necessary to reassign your local variable because it is modified in
             place.
    """

    for acronym, definition in new_acronyms.items():
        add_to_acronyms(acronyms, acronym, definition)

    return acronyms



def print_acronyms(acronyms):
    """
    Prints a acronym dictionary to the console.
    :param acronyms: The dictionary of acronyms.
    :return: None
    """

    print("\nACRONYM DICTIONARY")
    for k, v in acronyms.items():
        print("{0}:".format(k).ljust(10), v)



def check_first_letters(acronym, definition_tokens):
    """
    Checks to see if the words immediately before the acronym are a perfect match for the letters of the acronym.
    :param acronym: The acronym to be checked.
    :param definition_tokens: The tokens of the words to be checked against.
    :return: True if it matches, False if it does not.
    """

    if  len(acronym) != len(definition_tokens):
        return False

    for letter, definition_token in zip(acronym.upper(), definition_tokens):

        definition_token = definition_token.upper()

        if not definition_token.startswith(letter):
            if letter == "&" and definition_token == "AND":
                continue
            return False

    return True



def first_letter_search(acronym, acronym_i, n_letters, tokens):
    """
    Function for looking for the best definition for an acronym based on the first letter of the acronym.  Will search
    forwards and backwards for a length of UNMATCHED_ACRONYM_FIRSTLETTER_SEARCH_LEN for words that start with the first
    letter of the acronym.  It starts with the word that should have been the first based on the length of the acronym.
    IE if the acronym is 3 letters long, it starts 3 words backwards.
    :param acronym: The acronym to match the first letter of.
    :param acronym_i: The index of the acronym in the tokens (usually the sentence).
    :param n_letters: The length of the acronym.
    :param tokens: The tokens of the sentence (or doc or whatever).
    :return: False if no match is found.  Otherwise it returns a list of the tokens it believes is the match.
    """

    # sanity check: if it's at the beginning, we're out of luck
    if acronym_i == 0:
        return False

    found_match = False

    # search backwards for until the first letter of the token and first letter of the acronym match, or until
    # UNMATCHED_ACRONYM_FIRSTLETTER_SEARCH_LEN is reached
    for x in range(UNMATCHED_ACRONYM_FIRSTLETTER_SEARCH_LEN + 1):

        # search backwards: break if the first letter of this token is the first letter of the acronym
        backward_i = acronym_i - n_letters - x
        if backward_i - x >= 0:  # don't look past the 0 index
            if tokens[acronym_i - n_letters - x].upper()[0] == acronym[0]:
                found_match = True
                definition_tokens = tokens[(acronym_i - n_letters - x):acronym_i]
                break

        # search forwards: break if the first letter of this token is the first letter of the acronym
        forward_i = acronym_i - n_letters + x
        if forward_i < acronym_i and forward_i >= 0:  # only if it's at least 1 words ahead of the acronym
            if tokens[forward_i].upper()[0] == acronym[0]:
                found_match = True
                definition_tokens = tokens[forward_i:acronym_i]
                break


    if found_match:
        return definition_tokens
    else:
        return False



def capital_words_search(acronym_i, tokens):
    """
    Searches for the best match of the definition by looking for capitalized words.  Simply searches backwards up to
    UNMATCHED_ACRONYM_CAPITALS_SEARCH_LEN words far from the acronym
    :param acronym_i: The index of the acronym in the tokens.
    :param tokens: The word tokens (usually the sentence).
    :return: False if no match is found.  Otherwise it returns a list of tokens it believes is the definition.
    """

    # sanity check: if it's at the beginning, we're out of luck
    if acronym_i == 0:
        return False

    found_match = False
    tokens_start_i = 0

    # let's just take up to UNMATCHED_ACRONYM_CAPITALS_SEARCH_LEN capitalized words before
    for x in range(1, UNMATCHED_ACRONYM_CAPITALS_SEARCH_LEN + 1):

        token_i = acronym_i - x

        if tokens[token_i][0] in string.ascii_uppercase:
            found_match = True
            tokens_start_i = token_i
        else:
            if tokens[token_i] not in {"of", "and", "to", "the", "in"}:
                tokens_start_i = token_i + 1
                break

        # break if we reached the beginning of the sequence
        if token_i == 0:
            tokens_start_i = 0
            break


    if found_match:
        return tokens[tokens_start_i:acronym_i]
    else:
        return False



def parse_acronym(acronym, acronym_i, tokens):
    """
    Attempts to extract the definition of an acronym from a list of tokens (a sentence/doc).
    :param acronym: The acronym to be defined.
    :param acronym_i: The index of the acronym in the list of tokens.
    :param tokens: The list of tokens to search for the definition in (the sentence).
    :return: False if it doesn't pass the sanity checks.  Otherwise it will return the definition.
    """

    n_letters = len(acronym)


    # SANITY CHECKS
    # if it's at the beginning, we're out of luck
    if acronym_i == 0:
        return False
    # make sure it's more than one letter long
    if n_letters < 2:
        return False
    # make sure it's not too long
    if n_letters > MAX_ACRONYM_LEN:
        return False
    # make sure it contains at least one capital letter
    if re.match(r"[^A-Z]+", acronym):
        return False


    start_i = acronym_i - n_letters if acronym_i - n_letters > -1 else 0
    definition_tokens = tokens[start_i:acronym_i]

    # if it matches straight away, just return
    if check_first_letters(acronym, definition_tokens):
        return " ".join(definition_tokens)


    # search for the first letter in nearby words
    search_result = first_letter_search(acronym, acronym_i, n_letters, tokens)

    # couldn't find the first letter nearby, let's try just searching for capital words
    if not search_result:
        search_result = capital_words_search(acronym_i, tokens)

    if not search_result:
        print("WARNING -> Acronym does not match:", acronym, definition_tokens)
        # print("  !!!!! -> Could not find a good match, keeping original:", definition_tokens)
        return " ".join(definition_tokens)
    else:
        # print("    +ok -> Best match:", search_result)
        return " ".join(search_result)



def acronyms_from_sentence(sentence):
    """
    Extracts an acronym dictionary from a sentence.
    :param sentence: A sentence string (stripped of punctuation).
    :return: A dictionary of acronyms.
    """

    acronyms = {}
    tokens = split_tokens_soft(sentence)


    for i, token in enumerate(tokens):

        # matches strings enclosed in parentheses similar to "(*)"
        # acronym regex = lowercase letters, uppercase letters, numbers, "-", "&"
        match_obj = re.match(r"\(([a-zA-Z0-9\-&\']+)(?:\(s\))*\)", token)

        if match_obj:

            acronym = match_obj.group(1)
            acronym = acronym.replace("'s", "")  # remove bad grammar
            acronym = acronym.replace("(s)", "")  # remove bad grammar

            # skip strings that only have one capital letter at the beginning
            if re.match(r"[A-Z][a-z]{2,}", acronym):
                continue

            definition = parse_acronym(acronym, i, tokens)
            if definition:
                add_to_acronyms(acronyms, acronym, definition)

        else:
            match_obj = re.match(r"\((.+)\)", token)
            if match_obj:
                inside_parens = match_obj.group(1)
                if not re.match(r"[0-9\.%#\-]+|\.[a-zA-Z]{3,4}|[a-z]{2,5}\.", inside_parens): # numbers and some special chars OR file extensions
                    print("Unknown parentheses:", token)


    return acronyms



def acronyms_from_doc(doc):
    """
    Extracts an acronym dictionary from a document.  Splits it into strings and parses each individually.
    :param doc: A document as a string.
    :return: A dictionary of acronyms.
    """

    doc_acronyms = {}

    for sentence in split_sentences(doc):
        sentence_acronyms = acronyms_from_sentence(sentence)
        doc_acronyms.update(sentence_acronyms)

    return doc_acronyms





def main():
    """
    You know what it do...
    :return: None
    """

    with open("scratch.txt", "r") as f:
        doc = f.read()

    acronyms = acronyms_from_doc(doc)
    print_acronyms(acronyms)



if __name__ == "__main__":
    main()
