# Mark Evers
# 5/3/18
# acronyms.py
# Script for building a dictionary of acronyms.


from functools import reduce
import re
import string

# These constants are for when it has to search for the acronym
UNMATCHED_ACRONYM_PRE_SEARCH_LEN = 3  # how many words before the believed first word to search
UNMATCHED_ACRONYM_LEN = 7  # how many capitalized letters to include when above doesn't work
MAX_ACRONYM_LEN = 7  # any longer than this will be ignored



def add_to_acronyms(acronyms, acronym, definition):

    key = acronym
    x = 2

    while key in acronyms:

        if acronyms[key].lower() == definition.lower():
            break

        key = acronym + " ({0})".format(x)
        x += 1

    acronyms[key] = definition



def add_multiple_to_acronyms(acronyms, new_acronyms):

    for acronym, definition in new_acronyms.items():
        add_to_acronyms(acronyms, acronym, definition)



def print_acronyms(acronyms):

    print("\nACRONYM DICTIONARY")
    for k, v in acronyms.items():
        print("{0}:".format(k).ljust(10), v)



def first_letter_search(acronym, acronym_i, n_letters, tokens):

    found_match = False

    # search backwards for until the first letter of the token and first letter of the acronym match, or until UNMATCHED_ACRONYM_PRE_SEARCH_LEN is reached
    for x in range(1, UNMATCHED_ACRONYM_PRE_SEARCH_LEN + 1):

        # search backwards: break if the first letter of this token is the first letter of the acronym
        if tokens[acronym_i - n_letters - x].upper()[0] == acronym[0]:
            found_match = True
            definition_tokens = tokens[(acronym_i - n_letters - x):acronym_i]
            break

        # search forwards: break if the first letter of this token is the first letter of the acronym
        if acronym_i - n_letters + x < acronym_i - 2:  # only if it's at least 2 words ahead of the acronym
            if tokens[acronym_i - n_letters + x].upper()[0] == acronym[0]:
                found_match = True
                definition_tokens = tokens[(acronym_i - n_letters + x):acronym_i]
                break

        # break if we reached the beginning of the sentence
        if acronym_i - n_letters - x == 0:
            break

    if found_match:
        return definition_tokens
    else:
        return False



def capital_words_search(acronym_i, tokens):

    found_match = False
    tokens_start_i = 0

    # let's just take up to UNMATCHED_ACRONYM_LEN capitalized words before
    for x in range(1, UNMATCHED_ACRONYM_LEN + 1):

        if tokens[acronym_i - x][0] in string.ascii_uppercase:
            found_match = True
        else:
            tokens_start_i = acronym_i - x + 1
            break

        # break if we reached the beginning of the sequence
        if acronym_i - x == 0:
            tokens_start_i = 0
            break


    if found_match:
        return tokens[tokens_start_i:acronym_i]
    else:
        return False


def acronyms_from_doc(doc):

    doc_acronyms = {}
    sentences = re.split(r"[\.\?\!]\s+", doc)

    for sentence in sentences:
        sentence_acronyms = acronyms_from_sentence(sentence)
        doc_acronyms.update(sentence_acronyms)

    return doc_acronyms



def check_first_letters(acronym, definition_tokens):

    if  len(acronym) != len(definition_tokens):
        return False

    for letter, definition_token in zip(acronym.upper(), definition_tokens):

        definition_token = definition_token.upper()

        if not definition_token.startswith(letter):
            if letter == "&" and definition_token != "AND":
                return False

    return True



def parse_acronym(acronym, acronym_i, tokens):

    n_letters = len(acronym)

    # SANITY CHECKS
    # make sure it's more than one letter long
    if n_letters < 2:
        return False
    # make sure it's not too long
    if n_letters > MAX_ACRONYM_LEN:
        return False
    # make sure it contains at least one capital letter
    if re.match(r"[^A-Z]+", acronym):
        return False

    start_i = acronym_i - n_letters
    if start_i < 0:
        start_i = 0
    definition_tokens = tokens[start_i:acronym_i]

    # if it matches straight away, just return
    if len(definition_tokens) == n_letters:
        if reduce(lambda result, letter_enum: result and definition_tokens[letter_enum[0]].upper().startswith(letter_enum[1]), enumerate(acronym)):
            return string.capwords(" ".join(definition_tokens))

    # sanity check: make sure the acronym matches the first letter in each word
    if len(definition_tokens) < n_letters \
            or not reduce(
        lambda result, letter_enum: result and definition_tokens[letter_enum[0]].upper().startswith(letter_enum[1]),
        enumerate(acronym)):

        print("WARNING -> Acronym does not match:", acronym, definition_tokens)

        # search for the first letter in nearby words
        search_result = first_letter_search(acronym, acronym_i, n_letters, tokens)

        # couldn't find the first letter nearby, let's try just searching for capital words
        if not search_result:
            search_result = capital_words_search(acronym_i, tokens)

        if not search_result:
            print("  !!!!! -> Could not find a good match, keeping original:", definition_tokens)
        else:
            print("    +ok -> Best match:", search_result)
            definition_tokens = search_result

    definition = string.capwords(" ".join(definition_tokens))


def acronyms_from_sentence(sentence):

    acronyms = {}
    tokens = re.split(r"[\s/\-\+,\\\"]+", sentence)


    for i, token in enumerate(tokens):

        # matches strings enclosed in parentheses similar to "(*)"
        # acronym regex = lowercase letters, uppercase letters, numbers, "-", "&"
        match_obj = re.match(r"\(([a-zA-Z0-9\-&]+)\)$", token)

        if match_obj:

            acronym = match_obj.group(1)
            definition = parse_acronym(acronym, i, tokens)
            add_to_acronyms(acronyms, acronym, definition)


        else:
            match_obj = re.match(r"\((.+)\)", token)
            if match_obj:
                inside_parens = match_obj.group(1)
                print("Unknown parentheses:", token)


    return acronyms





def main():

    with open("scratch.txt", "r") as f:
        doc = f.read()

    acronyms = acronyms_from_doc(doc)
    print_acronyms(acronyms)



if __name__ == "__main__":
    main()
