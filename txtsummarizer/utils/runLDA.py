
import random

class runLDA():

    def __init__(self, document, vectorizer, k = 2, iterations = 500, alpha = 0.0001, eta = 0.0001):
        self.k = k
        self.iterations = iterations
        self.alpha = alpha
        self.document = document
        self.vectorizer = vectorizer

    def getWordTopicProportion(self):
        vocabulary = self.vectorizer.vocabulary_
        key_list = vocabulary.keys()
        val_list = vocabulary.values()
        wordTopic = {}
        randomTopic = {}
        
        for value, key in zip(val_list, key_list):
            for val in range(value):
                randomTopic[val] = self.randomTopic()

        print(randomTopic)

    def randomTopic(self):
        randomNumber = random.random()
        if randomNumber >= 0.5:
            return 0
        elif randomNumber < 0.5:
            return 1

    def assignedTopic(self, wordTopic):
        print(wordTopic)
            

        

