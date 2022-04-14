"""
Domain Model for Commits
"""


class Keyword:
    """
    Keyword associated with commit
    """

    keyword = ""
    positive_true = 0
    positive_false = 0
    unknown = 0

    def __init__(self, keyword: str):
        self.keyword = keyword

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Commit):
            return self.keyword == other.keyword
        return False

    def __hash__(self):
        return hash(self.keyword)

class Commit:
    """
    GitHub Commit
    """

    cmt_hash = ""
    author = ""
    date = ""
    text = ""
    fullString = ""

    def __init__(self, cmt_hash: str, author: str, date: str, text: str):
        self.cmt_hash = cmt_hash
        self.author = author
        self.date = date
        self.text = text
        self.fullString = ""

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Commit):
            return self.cmt_hash == other.cmt_hash
        return False

    def __hash__(self):
        return hash(self.cmt_hash)