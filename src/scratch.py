import re
from functools import reduce
import string

UNMATCHED_ACRONYM_PRE_SEARCH_LEN = 3
UNMATCHED_ACRONYM_LEN = 5
acronyms = {}


with open("scratch.txt", "r") as f:
    doc = f.read()


sentences = re.split(r"[\.\?\!]\s+", doc)

for sentence in sentences:

    tokens = sentence.split()

    for i, token in enumerate(tokens):

        # acronym regex = upper case letters, numbers, "-", "&"
        matchObj = re.match(r"\(([A-Z0-9\-&]+)\)", token)

        if matchObj:
            acronym = matchObj.group(1)
            # sanity check: make sure it contains at least one letter
            if re.match(r"[^A-Z]+", acronym):
                continue

            # sanity check: make sure it's more than one letter long
            n_letters = len(acronym)
            if n_letters < 2:
                continue

            definition_tokens = tokens[i - n_letters:i]

            # sanity check: make sure the acronym matches the first letter in each word
            if not reduce(lambda result, letter_enum: result and definition_tokens[letter_enum[0]].upper().startswith(letter_enum[1]), enumerate(acronym)):

                # it doesn't match... let's try some other stuff
                print("Acronym does not match:", acronym, definition_tokens)
                found_match = False

                # search backwards for until the first letter of the token and first letter of the acronym match, or until UNMATCHED_ACRONYM_PRE_SEARCH_LEN is reached
                for x in range(1, UNMATCHED_ACRONYM_PRE_SEARCH_LEN + 1):
                    # break if the first letter of this token is the first letter of the acronym
                    if tokens[i - n_letters - x].upper()[0] == acronym[0]:
                        found_match = True
                        definition_tokens = tokens[i - n_letters - x:i]
                        break

                    # break if we reached the beginning of the sentence
                    if i - n_letters - x == 0:
                        break


                if not found_match:

                    tokens_start_i = 0

                    # let's just take up to UNMATCHED_ACRONYM_LEN capitalized words before
                    for x in range(1, UNMATCHED_ACRONYM_LEN + 1):

                        if tokens[i - x][0] in string.ascii_uppercase:
                            found_match = True
                        else:
                            tokens_start_i = i - x + 1
                            break

                        # break if we reached the beginning of the sequence
                        if i - x == 0:
                            tokens_start_i = 0
                            break

                    if found_match:
                        definition_tokens = tokens[tokens_start_i:i]
                    else:
                        print("Could not find a good match...")


            definition = " ".join(definition_tokens)


            new_acronym = acronym
            x = 2
            while new_acronym in acronyms:
                if acronyms[new_acronym].lower() == definition.lower():
                    already_exists = True
                    break
                new_acronym = acronym + "({0})".format(x)
                x += 1

            if not already_exists:
                acronyms[new_acronym] = definition

        else:
            matchObj = re.match(r"\((.+)\)", token)
            if matchObj:
                inside_parens = matchObj.group(1)
                print("Unknown parentheses:", inside_parens)


print("Done!")
