"""
File Reader for Commits
"""
from typing import List

from model import Commit


class FileReader:
    """
    File reader parsing
    """

    def parse(self, file: str) -> List[Commit]:  # pylint: disable=R0201
        """"
        Loads all commits from a file.
        :rtype: object
        :param file: To be parsed.
        :return: All Commits in File.
        """
        f = open(file, "r", encoding='utf-8')
        commits = []
        c = None

        for line in f:
            if line.startswith("commit "):
                if c is not None:
                    c.text = c.text.strip().lower()
                    commits.append(c)
                c = Commit(cmt_hash="", author="", date="", text="")
                c.cmt_hash = line[7:].strip()
            elif line.startswith("Author: ") and c is not None:
                c.author = line[8:].strip()
            elif line.startswith("Date: ") and c is not None:
                c.date = line[6:].strip()
            elif line.startswith("http"):
                continue
            elif c is not None and line.strip():
                c.text += line.strip() + " "
            if c is not None:
                c.fullString += line
            

        if c is not None:
            c.text = c.text.strip().lower()
            commits.append(c)

        return commits

