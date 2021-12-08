import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from gensim import models, corpora
from gensim.models import Phrases
import numpy as np


wnl = WordNetLemmatizer()
lemmatizer = wnl.lemmatize

def tokenizer(document):
    text = re.sub('[^a-zA-Z]', ' ', document)
    tokens = text.lower().split()
    tokens = [lemmatizer(tkn) for tkn in tokens]
    return tokens


class TopicModel(object):

    def __init__(self, num_topics=100, min_word_count=20, 
                 top_most_common_words=10, min_doc_length=10, 
                 max_doc_length=50000, random_state=None):
        self.num_topics = num_topics
        self.min_word_count = min_word_count
        self.top_most_common_words = top_most_common_words
        
        assert max_doc_length > min_doc_length, \
               "max_doc_length must be greater than min_doc_length"
        self.min_doc_length = min_doc_length
        self.max_doc_length = max_doc_length
        self.random_state = random_state
        
        self.stop_words = self.getEnglishStopWords()
        self.bigramizer = Phrases()
        
    def fit(self, documents):
  
        self.tokens = self.preProcessCorpus(documents)
        self.dictionary = corpora.Dictionary(self.tokens)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.tokens]
        
        self.lda = self.getLDA(dictionary=self.dictionary, 
                               corpus=self.corpus, 
                               num_topics=self.num_topics, 
                               random_state=self.random_state)
        
        self.num_dominant_topics=min(10, self.num_topics)
        self.dominant_topic_ids = self.getDominantTopics(self.corpus, 
                                                         self.lda, 
                                                         self.num_dominant_topics)


    def __str__(self):
        description = ("topic model:\n\ttoken length = {0:,d}\n\tdictionary length = {1:,d}"
                       "\n\tnum_topics = {2:,d}\n\tmin_word_count = {3:,d}"
                       "\n\ttop_most_common_words = {4:,d}\n\tmin_doc_length = {5:,d}"
                       "\n\tmax_doc_length = {6:,d}")
        return description.format(len(self.tokens), 
                                  len(self.dictionary),
                                  self.num_topics, 
                                  self.min_word_count, 
                                  self.top_most_common_words, 
                                  self.min_doc_length, 
                                  self.max_doc_length)

    @staticmethod
    def getEnglishStopWords():
        
        stop_words = set(stopwords.words("english"))
        
        stop_words.add('please')
        stop_words.add('would')
        stop_words.add('use')
        stop_words.add('also')
        stop_words.add('thank')
        stop_words.add('sincerely')
        stop_words.add('regards')
        stop_words.add('hi')
        stop_words.add('hello')
        stop_words.add('greetings')
        stop_words.add('hey')
        stop_words.add('attachment')
        stop_words.add('attached')
        stop_words.add('attached_file')
        stop_words.add('see')
        stop_words.add('file')
        stop_words.add('comment')
        for item in 'abcdefghijklmnopqrstuvwxyz':
            stop_words.add(item)
        return stop_words
    
    
    @staticmethod
    def getFrequencies(tokens):
    
        frequencies = Counter()
        for row in tokens:
            frequencies.update(row)
        return frequencies
    
    @staticmethod
    def getLowFreqWords(frequencies, countCutOff):
       
        lowFreqTokens = set()
        for token, freq in frequencies.iteritems():
            if freq <= countCutOff:
                lowFreqTokens.add(token)
        return lowFreqTokens


    def preProcessCorpus(self, documents, min_word_count=None, 
                         top_most_common_words=None, min_doc_length=None, 
                         max_doc_length=None):
        if min_word_count is None:
            min_word_count = self.min_word_count
        if top_most_common_words is None:
            top_most_common_words = self.top_most_common_words
        if min_doc_length is None:
            min_doc_length = self.min_doc_length
        if max_doc_length is None:
            max_doc_length = self.max_doc_length
        
        tokens = tokenizer(documents)
        
        tokens = [tkn for tkn in tokens if len(tkn) < max_doc_length]
        
        self.bigramizer.add_vocab(tokens)
        
        tokens = [self.bigramizer[tokens]]
        
        tokens = [[t for t in tkn if t not in self.stop_words] for tkn in tokens]
        
        tokens = [tkn for tkn in tokens if len(tkn) > min_doc_length]
        
        # calculate token frequencies to exclude low and high frequency tokens
        # freqs = self.getFrequencies(tokens)
        
        # low_freq_tokens = set(x[0] for x in freqs.items() if x[1] < min_word_count)
        # high_freq_tokens = [word[0] for word in freqs.most_common(top_most_common_words)]
        
        # tokens =  [t for t in tokens if t not in low_freq_tokens]
        # tokens =  [t for t in tokens if t not in high_freq_tokens]
        
        # print '\nnumber of low frequency tokens pruned = {:,d}'\
        #       .format(len(low_freq_tokens))
        # print 'min_word_count = {:d}, top_most_common_words = {:,d}'\
        #       .format(min_word_count, top_most_common_words)
        # print 'number of high frequency tokens pruned = {:,d}'\
        #       .format(len(high_freq_tokens))
        # print 'tokens = {:,d} rows'.format(len(tokens))
        # print 'text pre-processing is complete\n'
        # print(tokens)
        return tokens


    def getLDA(self, dictionary=None, corpus=None, num_topics=None, 
               random_state=None):
        
        if dictionary is None:
            dictionary = self.dictionary
        if corpus is None:
            corpus = self.corpus
        if num_topics is None:
            num_topics = self.num_topics
        
        lda = models.ldamodel.LdaModel(corpus=corpus, 
                                       alpha='auto', 
                                       id2word=dictionary, 
                                       num_topics=num_topics,
                                       random_state=random_state)
        return lda


    def getDominantTopics(self, corpus, lda, num_dominant_topics=None):
        
        if corpus is None:
            corpus = self.corpus
        if lda is None:
            lda = self.lda
        if num_dominant_topics is None:
            num_dominant_topics = self.num_dominant_topics
        
        inference = lda.inference(corpus)
        inference = inference[0] 
        num_topics = lda.num_topics
        
       
        column_sum_of_weights = np.sum(inference, axis=0)
        sorted_weight_indices = np.argsort(column_sum_of_weights)
        idx = np.arange(num_topics - num_dominant_topics, num_topics)
        dominant_topic_ids = sorted_weight_indices[idx]
        
        
        dominant_topic_ids = dominant_topic_ids[::-1]
        
        
        return dominant_topic_ids.tolist()