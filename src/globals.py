# Mark Evers
# 5/9/2018
# globals.py
# Global variables and functions

from sys import stdout


DEBUG_LEVEL = 0


_PROGRESS_BAR_LAST_I = 100
def progress_bar(done, total, resolution = 0, text=""):
    """
    Prints a progress bar to stdout.
    :param done: Number of items complete
    :param total: Total number if items
    :param resolution: How often to update the progress bar (in percentage).  0 will update each time
    :param text: Text to display at the end.
    :return: None
    """

    global _PROGRESS_BAR_LAST_I

    # percentage done
    i = int(done / total * 100)
    if i == _PROGRESS_BAR_LAST_I and resolution:
        return

    # if it's some multiple of resolution
    if (not resolution) or (not i % resolution) or (i == 100):
        stdout.write('\r')
        # print the progress bar
        stdout.write("[{}]{}%".format(("-" * int(i / 2) + (">" if i < 100 else "")).ljust(50), str(i).rjust(4)))
        # print the text figures
        stdout.write("({}/{})".format(done, total).rjust(15))
        if text:
            stdout.write(" " + text)
        stdout.flush()

    if i == 100:
        # print("\n")
        stdout.write('\r')
        stdout.write(' ' * 80)
        stdout.write('\r')

    _PROGRESS_BAR_LAST_I = i


def debug(text, level = 0):

    if level <= DEBUG_LEVEL:
        print(text)
