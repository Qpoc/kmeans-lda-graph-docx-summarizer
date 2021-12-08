from .word_utils import WordUtils
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import pandas as pd
import re
import string
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

class LDA(WordUtils):

    def __init__(self, clusterNo, contents):
        self.clusterNo = clusterNo
        self.contents = contents

    def getClusterNo(self):
        return self.clusterNo
    
    def getContents(self):
        return self.clusterNo

    def vectorizedWords(self):

        cleanedContentText = []
        wordList = []

        content = self.contents
        cleanedText = self.getCleanedText(content.lower())
        cleanedContentText.append(cleanedText)

        vectorizer = CountVectorizer(stop_words={'english'})
        x = vectorizer.fit_transform(cleanedContentText)

        # df = pd.DataFrame(x.toarray, columns = vectorizer.get_feature_names())
        # print(df)
        # print(vectorizer.vocabulary_)

        return x, vectorizer