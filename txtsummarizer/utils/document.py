from .sentence_utils import SentenceUtils
from .word_utils import WordUtils
class Document(SentenceUtils, WordUtils):

    def __init__(self, title, content):
        self.title = title
        self.content = content

        SentenceUtils.__init__(self, title, content)
        WordUtils.__init__(self, title, content)
        
    def getTitle(self):
        return self.title

    def getContent(self):
        return self.content


