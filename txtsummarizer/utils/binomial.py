from enum import unique
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from numpy.linalg import norm
from nltk.stem import WordNetLemmatizer
import re
import random
import math

class Binomial():
    
    def __init__(self, content, isRemoveRedundant):
        self.content = content
        self.isRemoveRedundant = isRemoveRedundant

    def getBinomialDistribution(self):
        dictCleanedSentences, cleanedSentences = self.getListOfCleanedSentence()
        dictSentences, sentences = self.getListOfSentence()

        vectorizer = CountVectorizer()
        x = vectorizer.fit_transform(cleanedSentences)
        vectorizeSentence = x.toarray()

        nonRedundantSentenceList = []
        dictNonRedundantSentenceList = dict()

        # testing for similar sentences, uncomment this to compare sentence 1 and sentence 2
        # print(vectorizeSentence[0], vectorizeSentence[1])
        # pnt(np.inner(verictorizeSentence[0],vectorizeSentence[1])/(norm(vectorizeSentence[0]) * norm(vectorizeSentence[1])))

        redundantSentences = []

        # Compare sentences if similar
        for index in range(len(sentences)):
            if self.isRemoveRedundant:
                if not self.getSentenceSimilarity(vectorizeSentence[index], nonRedundantSentenceList):
                    nonRedundantSentenceList.append(vectorizeSentence[index])
                    dictNonRedundantSentenceList[index] = vectorizeSentence[index]
                else:
                    redundantSentences.append("<p><span class='text-secondary'>Redundant Sentence: </span> " + str(dictSentences[index]) + " </p>")
            else:
                nonRedundantSentenceList.append(vectorizeSentence[index]) 
                dictNonRedundantSentenceList[index] = vectorizeSentence[index]
        
        # print(redundantSentences)

        intersections = dictNonRedundantSentenceList.keys() & dictSentences.keys()

        newSentences = []
        dictNewSentences = dict()
        newCleanedSentences = []
        dictNewCleanedSentences = dict()

        # get the original sentence and cleaned sentence based on the intersection of keys between NonRedundantSentence and sentences

        for intersection in intersections:
            newSentences.append(dictSentences[intersection])
            dictNewSentences[intersection] = dictSentences[intersection]
            newCleanedSentences.append(cleanedSentences[intersection])
            dictNewCleanedSentences[intersection] = cleanedSentences[intersection]

        # Tokenize new sentence
        tokens = []
        N = len(newSentences)
        dictTokens = dict()
        bigramTokens = []
        dictBigramTokens = dict()

        for keys in dictNewCleanedSentences.keys():
            tokens.append(word_tokenize(dictNewCleanedSentences[keys]))
            dictTokens[keys] = word_tokenize(dictNewCleanedSentences[keys])
            bigramTokens.append(list(nltk.bigrams(word_tokenize(dictNewCleanedSentences[keys]))))
            dictBigramTokens[keys] = list(nltk.bigrams(word_tokenize(dictNewCleanedSentences[keys])))

        # documentTokens
        
        # getUniqueWords
        uniqueWords = []
        for perSentenceToken in tokens:
            for token in perSentenceToken:
                uniqueWords.append(token)

        uniqueWords = set(uniqueWords)
        uniqueWordsOccurences = dict()

        # get occurences of uniqueWords
        for word in uniqueWords:
            index = 0
            for sentence in newSentences:
                if word in sentence:
                    index += 1
                    uniqueWordsOccurences[word] = index

        # get probability of each terms
        Pi = dict()
        for keys in uniqueWordsOccurences.keys():
            Pi[keys] = int(uniqueWordsOccurences[keys])/N
        
        Nij = dict()
        Nj = Pi
        # get co-occurences
        for keys in dictBigramTokens.keys():
            for sentenceBigram in dictBigramTokens[keys]:
                index = 0
                for bigram in dictBigramTokens.values():
                    if sentenceBigram in bigram:
                        index += 1
                        Nij[sentenceBigram] = index

        Pnij = dict()
        lexicalAssoc = dict()

        for uniqueWordKeys in Nj.keys():
            probabilityBigramTerm = []
            for bigramKeys in Nij.keys():
                binomialDist = binompmf(Nij[bigramKeys], N, Nj[uniqueWordKeys])
                probabilityBigramTerm.append(binomialDist)
            Pnij[uniqueWordKeys] = probabilityBigramTerm

        # print(Pnij)

        Inf = dict()
        for keys in Pnij.keys():
            value = []
            for pnij in Pnij[keys]:
                if pnij == 0.0:
                    pnij = 0.000001
                inf = -1 * math.log2(float(pnij))
                value.append(inf)
            Inf[keys] = value
        
        # print(Inf)
        
        termWeights = dict()

        for keys in Inf.keys():
            termWeight = 0.0
            for weight in Inf[keys]:
                termWeight += weight
            termWeights[keys] = termWeight/len(uniqueWords)
        
        # Calculating sentence weight
        sentenceWeights = dict()
        for sentenceKeys in dictNewCleanedSentences.keys():
            sentenceWeight = 0.0
            for termKeys in termWeights.keys():
                sentenceWeight += (float(termWeights[termKeys]) * int(self.countSubstring(dictNewCleanedSentences[sentenceKeys], termKeys)))
            sentenceWeights[sentenceKeys] = sentenceWeight

      
        sortedSentenceWeights = dict(sorted(sentenceWeights.items(), key=lambda item: item[1], reverse=True))

        # print(sortedSentenceWeights)
        summary = ""
        listSummary = []
        for sentenceKey in sortedSentenceWeights.keys():
            summary += ''.join(dictNewSentences[sentenceKey])
            summary += " "
        
        listSummary.append(summary)

        # print(sortedSentenceWeights)
        displaySentenceWeight = []
        for keys in sortedSentenceWeights.keys():   
            displaySentenceWeight.append(str(dictNewSentences[keys]) + "\n<span class='text-secondary fst-italic fs-bold'>Weight: " + str(sortedSentenceWeights[keys]) + "</span>")

        return listSummary, displaySentenceWeight, redundantSentences
        # binompmf(Nij,N,Nj)

    def countSubstring(self, string, sub_string):

        length = len(string)
        counter = 0
        for i in range(length):
            for j in range(length):
                if string[i:j+1] == sub_string:
                    counter +=1
        return counter


    def getKeysFromDict(self, bigram, dictBigramTokens):
        for keys in dictBigramTokens:
            if bigram in dictBigramTokens[keys]:
                return keys

    def getListOfSentence(self):
        index = 0
        dictSentences = dict()
        for sentence in sent_tokenize(self.content):
            dictSentences[index] = sentence
            index += 1

        return dictSentences, sent_tokenize(self.content)

    # get sentence
    def getListOfCleanedSentence(self):
        lemmatizer = WordNetLemmatizer()
        dictCleanedSentences = dict()
        cleanedSentences = []
        index = 0
        for sentence in sent_tokenize(self.content):
            cleanedSentence = ' '.join([lemmatizer.lemmatize(word.lower()) for word in word_tokenize(re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|[0-9]|http.+?", "", sentence)) if len(word.lower()) > 3 and word.lower() not in stopwords.words('english') and word not in ('!', '.', ',', '/', '\\', ',', ':',
                               ';', '?', '.', '[', ']', '(', ')', '-', '*', '“', '”', '—' , '"', '\’', '\n', '\r', '\t') and not word.lower().isdecimal()]) 
            dictCleanedSentences[index] = cleanedSentence
            cleanedSentences.append(cleanedSentence)
            index += 1
        return dictCleanedSentences, cleanedSentences

    def getSentenceSimilarity(self, vector1, vector2):
        for nonRedundantList in vector2:
            similarityWeight = np.inner(vector1,nonRedundantList)/(norm(vector1) * norm(nonRedundantList))
            if similarityWeight > 0.66:
                return True
        return False

def factorial(n):
    x = 1
    for i in range(1, n+1):
        x *= i
    return x

def combination(n, k):
    return factorial(n)/(factorial(k)*factorial(n-k))

def binompmf(k,n,p):
    return combination(n,k)*(p**k)*((1-p)**(n-k))