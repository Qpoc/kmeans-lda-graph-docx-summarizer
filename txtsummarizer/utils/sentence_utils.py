
class SentenceUtils:

    def __init__(self, title, content):
        self.title = title
        self.content = content
        
    def getNumberOfSentence(self):
        return self.content.count(".")

    def getListOfSentence(self):
        return self.content.split(".")
        