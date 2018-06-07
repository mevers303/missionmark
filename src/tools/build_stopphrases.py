from data import get_connection
import re




def main():

    conn = get_connection()

    query = """
                SELECT buying_organizations
                FROM import.govwin_opportunity
            """

    with conn.cursor() as cursor:
        cursor.execute(query)
        phrases = {org["name"] for row, in cursor for org in row}

    expr = re.compile(r"\(.+\)|CODE [A-Z0-9]+|\-")

    with open("../../stopwords_phrases.txt", "a") as f:

        for phrase in phrases:

            if " " not in phrase:
                phrase = "DEPARTMENT OF " + phrase

            spl = expr.split(phrase)
            for item in spl:

                item = item.strip()
                if not item:
                    continue



                f.write(item + "\n")
                print(item)




if __name__ == "__main__":
    main()
