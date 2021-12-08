from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
import re
import string
import nltk

nltk.download('stopwords')
nltk.download('wordnet')


class WordUtils:

    def __init__(self, titles, contents):
        self.titles = titles
        self.contents = contents

    def vectorizedWords(self):

        cleanedContentText = []
        wordList = []

        for content in self.contents:
            cleanedText = self.getListOfCleanedSentence(content.lower())
            cleanedContentText.append(cleanedText)

        vectorizer = TfidfVectorizer(stop_words={'english'}, norm='l2', use_idf=True, smooth_idf=False)
        x = vectorizer.fit_transform(cleanedContentText)

        return x.toarray()

    def getListOfCleanedSentence(self, content):
        lemmatizer = WordNetLemmatizer()
        dictCleanedSentences = dict()
        cleanedSentences = ""
        index = 0
        for sentence in sent_tokenize(content):
            cleanedSentences += ' '.join([lemmatizer.lemmatize(word.lower()) for word in word_tokenize(re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|[0-9]|http.+?", "", sentence)) if len(word.lower()) > 3 and word.lower() not in stopwords.words('english') and word not in ('!', '.', ',', '/', '\\', ',', ':',
                               ';', '?', '.', '[', ']', '(', ')', '-', '*', '“', '”', '—' , '"', '\’', '\n', '\r', '\t') and not word.lower().isdecimal()]) 
            cleanedSentences += " "
            dictCleanedSentences[index] = cleanedSentences
            index += 1
        return cleanedSentences

    def getCleanedText(self, content):
        returnSentence = []
        word_list = []
        sentences = content.split(".")

        ps = PorterStemmer()
        lemmatizer = WordNetLemmatizer()

        for sentence in sentences:
            sentence = ''.join([i for i in sentence if not i.isdigit()])
            sentence = ''.join([l for l in sentence if l not in ('!', '.', ',', '/', '\\', ',', ':',
                               ';', '?', '.', '[', ']', '(', ')', '-', '*', '“', '”', '"', '\’', '\n', '\r', '\t')])
            sentence = sentence.strip()
            sentence = [lemmatizer.lemmatize(word) for word in sentence.split(
            ) if word not in stopwords.words('english')]

            returnSentence.append(sentence)

        for sentence in returnSentence:
            for word in sentence:
                word_list.append(word)

        listToStr = ' '.join([str(elem) for elem in word_list])

        return listToStr

    def getListOfWords(self):
        return self.content.split(" ")

    def getNumberOfWords(self):
        return len(self.content.split(" "))

    def getUniqueWords(self):
        uniqueWords = set(self.content.split(" "))
        return uniqueWords
