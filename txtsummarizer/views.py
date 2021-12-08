from django.shortcuts import render

from .utils.lda import LDA

from .models import News
from .utils.document import Document
from .utils.sentence_utils import SentenceUtils
from .utils.bisectingKMeans import  bisectingKMeans, kMeans
from .utils.binomialDistribution import BinomialUtils
from .utils.runLDA import runLDA
from .utils.topicModel import TopicModel
from .utils.documentSummaries import DocumentSummaries
from .utils.binomial import Binomial
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

import docx2txt
import math
from numpy import *
import numpy as np
import pandas as pd
from django.views.decorators.csrf import csrf_protect

# Create your views here.
@csrf_protect
def index(request):
    
    if request.method == 'POST':
        files = []
        if len(request.FILES.getlist('doc_files')) > 0:
            index = 0
            for file in request.FILES.getlist('doc_files'):
                if index <= 0:
                    navItem = '<li class="nav-item" role="presentation"> <button class="nav-link active" id="pills-tab-' + str(index) + '" data-bs-toggle="pill" data-bs-target="#pills-' + str(index) + '" type="button" role="tab" aria-controls="pills-home" aria-selected="true" >' + file.name + '</button></li>'

                    tabPane = '<div class="tab-pane fade show active" id="pills-' + str(index) + '"role="tabpanel" aria-labelledby="pills-tab-' + str(index) + '"><textarea class="form-control" name="pills-textarea" cols="30"rows="10">' + docx2txt.process(file) + '</textarea></div>'
                else:
                    navItem = '<li class="nav-item" role="presentation"> <button class="nav-link" id="pills-tab-' + str(index) + '" data-bs-toggle="pill" data-bs-target="#pills-' + str(index) + '" type="button" role="tab" aria-controls="pills-home" aria-selected="true" >' + file.name + '</button></li>'

                    tabPane = '<div class="tab-pane fade" id="pills-' + str(index) + '"role="tabpanel" aria-labelledby="pills-tab-' + str(index) + '"><textarea class="form-control" name="pills-textarea" cols="30" rows="10">' + docx2txt.process(file) + '</textarea></div>'

                files.append({
                    "file_names" : file.name,
                    "nav_items" : navItem,
                    "tab_panes" : tabPane,
                })

                index += 1

            context = {
                "payload" : files,
                "length" : len(request.FILES.getlist('doc_files'))
            }
        elif len(request.POST.getlist('pasted-textarea')) > 0:
            summaries = []
            termWeights = []
            content = request.POST['pasted-textarea']

            tokens = word_tokenize(content)
            sentences = sent_tokenize(content)

            if int(len(tokens)) <= 20:
                context = {
                    "status" : "error",
                    "message" : "<p>Please enter enough text to summarize.</p>"
                }

                return render(request, 'txtsummarizer/index.html', context)

            topicModel = TopicModel(num_topics = 1)
            topicModel.fit(content)
            docSummaries = DocumentSummaries(topicModel, num_dominant_topics=1, number_of_sentences= int(request.POST['numberSentence']))
            docSummaries.summarize(content)
            
            summary, termWeight = docSummaries.getSummary()

            strRemoveRedundant = request.POST['removeRedundant']
            isRemoveRedundant = True
            if strRemoveRedundant == "False":
                isRemoveRedundant = False

            sortedSentenceWeights = []
            redundantSentences = []

            binomial = Binomial(summary[0], isRemoveRedundant)
            refinedSummary, sortedSentenceWeight, redundantSentence = binomial.getBinomialDistribution()
            # objBino = BinomialUtils("test",summary[0])
            # refinedSummary = objBino.getBinomialDistribution()
            summaries.append(refinedSummary)
            termWeights.append(termWeight)
            sortedSentenceWeights.append(sortedSentenceWeight)
            redundantSentences.append(redundantSentence)
                
            context = {
                "listSummary" : summaries,
                "termWeights" : termWeights,
                "sentenceWeights" : sortedSentenceWeights,
                "redundantSentences" : redundantSentences
            }

        elif len(request.POST.getlist('pills-textarea')) > 0:
            titles = []
            contents = []
            index = 0

            for content in request.POST.getlist('pills-textarea'):
                titles.append(str(index))
                contents.append(content)
                index += 1

            document = Document(titles, contents).vectorizedWords()

            numberCluster = 2
           
            if request.POST['numberCluster'] == 'auto':
                numberCluster = int(np.sqrt(math.ceil(len(contents)/2)))
            else:
                numberCluster = int(request.POST['numberCluster'])
            
            if len(request.POST.getlist('pills-textarea')) < numberCluster:
                numberCluster = len(request.POST.getlist('pills-textarea'))
            
            # print(numberCluster)

            labels = np.array(titles)
            predict = bisectingKMeans(document, numberCluster, 200)

            count = 1
            clusters = []
            clusteredDocuments = []
            dictClusteredDocuments = dict()
            for index in range(len(predict[:, 0])):
                # print("Cluster " + str(predict[index, 0]) + ": Document title: " + str(labels[index]))
                dictClusteredDocuments[str(labels[index])] = str(predict[index, 0])
                clusters.append(predict[index, 0])
            
            # print(dictClusteredDocuments)

            uniqueCluster = set(clusters)

            clusterContents = []
            clusterNoDocuNo = []
            for i in range(len(uniqueCluster)):  
                content = ""
                for j in range(len(predict[:, 0])):
                    if predict[j, 0] == i:
                        content += contents[j]
                        clusterNoDocuNo.append(labels[j])

                clusteredDocuments.append(clusterNoDocuNo)
                clusterContents.append(content)

            # print(clusteredDocuments)

            summaries = []
            termWeights = []

            strRemoveRedundant = request.POST['removeRedundant']
            isRemoveRedundant = True
            if strRemoveRedundant == "False":
                isRemoveRedundant = False

            sortedSentenceWeights = []
            redundantSentences = []

            for content in clusterContents:
                topicModel = TopicModel(num_topics = 1)
                topicModel.fit(content)
                docSummaries = DocumentSummaries(topicModel, num_dominant_topics=1, number_of_sentences= int(request.POST['numberSentence']))
                docSummaries.summarize(content)
                summary, termWeight = docSummaries.getSummary()
                binomial = Binomial(summary[0], isRemoveRedundant)
                refinedSummary, sortedSentenceWeight, redundantSentence = binomial.getBinomialDistribution()
                # objBino = BinomialUtils("test",summary[0])
                # refinedSummary = objBino.getBinomialDistribution()
                summaries.append(refinedSummary)
                termWeights.append(termWeight)
                sortedSentenceWeights.append(sortedSentenceWeight)
                redundantSentences.append(redundantSentence)
                
            context = {
                "listSummary" : summaries,
                "termWeights" : termWeights,
                "sentenceWeights" : sortedSentenceWeights,
                "redundantSentences" : redundantSentences,
                "clusteredDocs" : dictClusteredDocuments
            }

            # print(context)
            
            return render(request, 'txtsummarizer/index.html', context)
        else:
            context = {
                "status" : "An error occurred while processing the request"
            }

        return render(request, 'txtsummarizer/index.html', context)
    else:
        return render(request, 'txtsummarizer/index.html')

def jake_view(request):

   # documnets = Document("Coronavirus disease (COVID-19)", "Most people infected with the virus will experience mild to moderate respiratory illness and recover without requiring special treatment.However, some will become seriously ill and require medical attention. Older people and those with underlying medical conditions like cardiovascular disease, diabetes, chronic respiratory disease, or cancer are more likely to develop serious illness. Anyone can get sick with COVID-19 and become seriously ill or die at any age. ")
    ObjBinomial = BinomialUtils("Coronavirus disease (COVID-19)",'SUMMARY: House committee on health chairperson and Quezon Rep. Angelina Tan said that the health system capacity is “at the brink of a collapse.” HEADLINE: SOS! Need help! Lawmaker warns PH health system on brink of collapse due to COVID-19 By Billy Begas House committee on health chairperson and Quezon Rep. Angelina Tan said that the health system capacity is “at the brink of a collapse.”Tan made the statement during her sponsorship speech at the House Committee on Ways and Means for the proposed Health Procurement and Stockpiling Act. “We are undergoing through a huge surge from the first COVID-19 case we had from last year but this time with more variants and faster transmission rate. We are already at the brink of a collapse of our health system capacity with more COVID-19 patients coming in, more infected healthcare workers, and lack of hospital beds,” Tan said. Aside from the scarcity of vaccines, Tan said the supply of medicines that could be used to treat COVID-19 patients are declining due to high demand.“To add more to our challenges, we have also been receiving reports of hoarding COVID-19 vaccines and raw materials for its production, especially when the US invoked its National Defense Production Act, which would definitely shake the steady supply of vaccines around the globe,” Tan added.She said the proposal will prevent the repeat of what happened last year when there was scarcity in the supply of facemask, alcohol, personal protective equipment, mechanical ventilators and RT-PCR kits and machines.“To avoid a similar situation, I have filed this bill with the main thrust of protecting the public health and safety by preventing and controlling the spread of diseases and other health hazard through stockpiling of essential and critical drugs and medicines, vaccines, devices and materials to effectively and swiftly confront the devastating consequences of public health emergency,” she added. The tax provision of the unnumbered substitute bill was approved by the Ways and Means committee.')
    
    
    # print( ObjBinomial.getBinomialDistribution(ObjBinomial.getStopWords()))
    
    binomial = ObjBinomial.getBinomialDistribution()
    
    # internalInfo = {
    #    "binomial" : binomialDist

        
    # }


    return render(request, 'txtsummarizer/jake.html', {})


def jheymie_view(request):
    return render(request, 'txtsummarizer/jheymie.html', {})


def renwell_view(request):
    return render(request, 'txtsummarizer/renwell.html', {})


def cyrus_view(request):

    documents = Document("Coronavirus disease (COVID-19)", "Most people infected with the virus will experience mild to moderate respiratory illness and recover without requiring special treatment.However, some will become seriously ill and require medical attention. Older people and those with underlying medical conditions like cardiovascular disease, diabetes, chronic respiratory disease, or cancer are more likely to develop serious illness. Anyone can get sick with COVID-19 and become seriously ill or die at any age. ")

    uniqueWords = documents.getUniqueWords()
    documentContent = documents.getContent()
    documentTitle = documents.getTitle()
    numberOfSentence = documents.getNumberOfSentence()
    wordList = documents.getListOfWords()
    numWords = documents.getNumberOfWords()

    context = {
        "document_title" : documentTitle,
        "content" : documentContent,
        "num_sentence" : numberOfSentence,
        "unique_words": uniqueWords,
        "list_words" : wordList,
        "num_words" : numWords
    }

    return render(request, 'txtsummarizer/cyrus.html', context)
