from logging import log

from numpy.lib.function_base import append
from .topicModel import tokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk.data
import numpy as np

nltk.download('punkt')

class topicSummary(object):

    def __init__(self, topic_id, terms, weights, sentences):
        self.topic_id = topic_id
        self.terms = terms
        self.weights = weights
        self.sentences = sentences

    def __str__(self):
        if self.sentences is None or len(self.sentences) == 0:
            return 'topic does not have any sentences'
        text = str()
        
        for t in self.terms:
            text += '{:s},'.format(t)
        text += '\n'
        
        for w in self.weights:
            text += '{:5.4f},'.format(w)
        text += '\n'
        for sentence in self.sentences:
            text += sentence[2] + ' '
        return text



def innerProduct(bow1, bow2):
    keys1 = set(bow1)
    keys2 = set(bow2)
    keys = keys1.intersection(keys2)
    if not keys:
        return 0.0
    inner_product = 0.0
    for key in keys:
        inner_product += bow1[key] * bow2[key]
    sum1 = 0.0
    sum2 = 0.0
    for v in bow1.values():
        sum1 += v*v
    for v in bow2.values():
        sum2 += v*v
    inner_product /= np.sqrt(sum1 * sum2)

    return inner_product
    

def cosineSimilarity(sentence_bow, list_of_sentence_bow):
    
    for bow in list_of_sentence_bow:
        inner_product = innerProduct(sentence_bow, bow)
        if inner_product >= 0.66:
            return True
    return False



class DocumentSummaries(object):
    
    def __init__(self, model, num_dominant_topics=5, number_of_sentences=5):
     
        self.num_dominant_topics = num_dominant_topics
        self.number_of_sentences= number_of_sentences
        self.lda = model.lda
        self.dictionary = model.dictionary
        self.bigramizer = model.bigramizer
    
    
    def summarize(self, documents):

        lemmatizer = WordNetLemmatizer()

        strToken = []
        tokens = [tokenizer(documents)]
        for token in tokens:
            for tkn in token:
                if len(tkn) > 3 and tkn not in ('!', '.', ',', '/', '\\', ',', ':',
                               ';', '?', '.', '[', ']', '(', ')', '-', '*', '“', '”', '"', '\’', '\n', '\r', '\t') and tkn not in stopwords.words('english'):
                    strToken.append(lemmatizer.lemmatize(tkn))
        tokens = []
        tokens.append(strToken)

        tokens = [tkn for tkn in tokens]
    
        corpus = [self.dictionary.doc2bow(tkn) for tkn in tokens]
        
        self.dominant_topic_ids = self.getDominantTopics(corpus)
       
        self.sentence_groups = self.splitIntoSentences(documents)

        self.distributions = self.getSentenceDistributions()
       
        self.summary_data = self.sentenceSelection(verbose=False)
    
    def getDominantTopics(self, corpus):
    
        
        inference = self.lda.inference(corpus)
        inference = inference[0]
        num_topics = self.lda.num_topics
        

        column_sum_of_weights = np.sum(inference, axis=0)
        sorted_weight_indices = np.argsort(column_sum_of_weights)
        idx = np.arange(num_topics - self.num_dominant_topics, num_topics)
        dominant_topic_ids = sorted_weight_indices[idx]
    
        dominant_topic_ids = dominant_topic_ids[::-1]
        
        return dominant_topic_ids.tolist()

    
    def splitIntoSentences(self, documents, MIN_SENTENCE_LENGTH = 8, MAX_SENTENCE_LENGTH = 25):
       
        sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')
  
        sentence_groups = list()
        
        sentences = sentence_detector.tokenize(documents)
        sentence_group = list()
        for k, sentence in enumerate(sentences):
            length = len(sentence.split())
            if (length > MIN_SENTENCE_LENGTH and length < MAX_SENTENCE_LENGTH):
                sentence_group.append((k, sentence))
        sentence_groups.append(sentence_group)
        return sentence_groups
    
    
    def getSentenceDistributions(self):
        distributions = list()
        get_bow = self.dictionary.doc2bow
        get_document_topics = self.lda.get_document_topics
        for sentences in self.sentence_groups:
            sentence_distributions = list()
            for k, sentence in sentences:
                tkns = tokenizer(sentence)
                if tkns is None:
                    continue
                bow = get_bow(tkns)
                dist = get_document_topics(bow)
                try:
                    dist = max(dist, key=lambda x: x[1])
                except ValueError:
                    continue
                sentence_distributions.append((k, dist))
            distributions.append(sentence_distributions)
        return distributions
    
    
    def sentenceSelection(self, verbose=False):
        
        results_per_docket = dict()
        results_per_docket['number_of_documents'] = len(self.sentence_groups)
        results_per_docket['dominant_topic_ids'] = self.dominant_topic_ids
    
        for dtid in self.dominant_topic_ids:
            results_per_topic = dict()
            
            top_sentences = self.sentencesPerTopic(dtid)
            topic_terms = self.lda.show_topic(dtid)
            
            terms = [t[0] for t in topic_terms]
            weights = [w[1] for w in topic_terms]
            
            ts = topicSummary(topic_id = dtid, terms=terms, 
                              weights=weights, sentences=top_sentences)
            
            if verbose:
                displaySummary(top_sentences, topic_terms)
            
            results_per_docket[dtid] = ts
        
        return results_per_docket


    def sentencesPerTopic(self, dominant_topic_id):
        
        filtered_by_topic_id = self.filterSentencesByTopic(dominant_topic_id)
        
        if len(filtered_by_topic_id) == 0:
            return
        
        sn = 0
        
        similarity_list = list()
        top_sentences = list()
        
        passage = self.sentence_groups[0]
        
        for sentence in passage:
            if len(top_sentences) >= self.number_of_sentences:
                break
            sentence_bow = self.dictionary.doc2bow(tkns for tkns in sentence[1].lower().split())
            sentence_bow = dict(sentence_bow)

            # if not cosineSimilarity(sentence_bow, similarity_list):
            #     similarity_list.append(sentence_bow)
            top_sentences.append((0, sentence[1]))

        return top_sentences
    
    def filterSentencesByTopic(self, topic_id):
        
        filtered_by_topic_id = list()
        for k, distribution in enumerate(self.distributions):
            filtered = [d for d in distribution if d[1][0] == topic_id]
            for item in filtered:
                filtered_by_topic_id.append((k, item[0], item[1][1]))
        return filtered_by_topic_id
    
    
    def getSummary(self):

        paragraph = []
        termWeight = []
        summarizeSentences = ""

        # print('The dominant topics in descending order are:')
        # for dtid in self.dominant_topic_ids:
        #     print(dtid), 
        # print('')
        
        for k in range(self.num_dominant_topics):
            dtid = self.dominant_topic_ids[k]
        
            topicSummary = self.summary_data[dtid]
            terms = topicSummary.terms
            weights = topicSummary.weights
            num_terms = len(terms)
            sentences = topicSummary.sentences
            
            # print('\nTopic {:d}'.format(dtid))
            # print('The top {:d} terms and corresponding weights are:'.format(num_terms))
            for term, weight in zip(terms, weights):
                # print(' * {:s} ({:5.4f})'.format(term, weight))
                termWeight.append('{:s} ({:5.4f})'.format(term, weight))
            
            # print('\n\nThe selected sentences are:'),
            if sentences is not None:
                n_sentences = len(sentences)
            # for j in range(n_sentences):
            #     item = sentences[j]
            #     print(item)
            # print('-----Summary-----')
                for j in range(n_sentences):
                    item = sentences[j]
                    sentence = item[1]
                    summarizeSentences += sentence + " "
                paragraph.append(summarizeSentences)
            # print(' ')
        return paragraph, termWeight