# Mark Evers
# 5/3/18
# acronym_builder.py
# Script for building a dictionary of acronyms.


from functools import reduce
import re
import string

# These constants are for when it has to search for the acronym
UNMATCHED_ACRONYM_PRE_SEARCH_LEN = 3  # how many words before the believed first word to search
UNMATCHED_ACRONYM_LEN = 5  # how many capitalized letters to include when above doesn't work



def add_to_acronyms(acronyms, acronym, definition):

    key = acronym
    x = 2

    while key in acronyms:

        if acronyms[key].lower() == definition.lower():
            break

        key = acronym + " ({0})".format(x)
        x += 1

    acronyms[key] = definition



def first_letter_search(acronym, acronym_i, n_letters, tokens):

    found_match = False

    # search backwards for until the first letter of the token and first letter of the acronym match, or until UNMATCHED_ACRONYM_PRE_SEARCH_LEN is reached
    for x in range(1, UNMATCHED_ACRONYM_PRE_SEARCH_LEN + 1):

        # break if the first letter of this token is the first letter of the acronym
        if tokens[acronym_i - n_letters - x].upper()[0] == acronym[0]:
            found_match = True
            definition_tokens = tokens[acronym_i - n_letters - x:acronym_i]
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



def acronyms_from_sentence(sentence):

    acronyms = {}
    tokens = sentence.split()


    for i, token in enumerate(tokens):

        # matches strings enclosed in parentheses similar to "(*)"
        # acronym regex = upper case letters, numbers, "-", "&"
        matchObj = re.match(r"\(([A-Z0-9\-&]+)\)", token)

        if matchObj:

            acronym = matchObj.group(1)
            n_letters = len(acronym)

            # sanity check: make sure it's more than one letter long
            if n_letters < 2:
                continue

            # sanity check: make sure it contains at least one letter
            if re.match(r"[^A-Z]+", acronym):
                continue


            definition_tokens = tokens[i - n_letters:i]

            # sanity check: make sure the acronym matches the first letter in each word
            if not reduce(lambda result, letter_enum: result and definition_tokens[letter_enum[0]].upper().startswith(letter_enum[1]), enumerate(acronym)):

                print("WARNING -> Acronym does not match:", acronym, definition_tokens)

                # search for the first letter in nearby words
                search_result = first_letter_search(acronym, i, n_letters, tokens)

                # couldn't find the first letter nearby, let's try just searching for capital words
                if not search_result:
                    search_result = capital_words_search(i, tokens)

                if not search_result:
                    print("    bad -> Could not find a good match...")
                else:
                    print("     ok -> Best match:", search_result)
                    definition_tokens = search_result


            definition = string.capwords(" ".join(definition_tokens))
            add_to_acronyms(acronyms, acronym, definition)


        else:
            matchObj = re.match(r"\((.+)\)", token)
            if matchObj:
                inside_parens = matchObj.group(1)
                print("Unknown parentheses:", inside_parens)


    return acronyms




def main():

    with open("scratch.txt", "r") as f:
        doc = f.read()

    acronyms = acronyms_from_doc(doc)

    print("\nACRONYM DICTIONARY")  # newline
    for k, v in acronyms.items():
        print("{0}:".format(k).ljust(10), v)


if __name__ == "__main__":
    main()
