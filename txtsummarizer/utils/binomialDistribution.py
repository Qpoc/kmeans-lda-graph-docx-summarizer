from numpy.lib.function_base import append
from .sentence_utils import SentenceUtils
from .word_utils import WordUtils
import numpy as np
from numpy import product, random
import nltk
import math as mt
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from numpy.linalg import norm
from itertools import product
from scipy import spatial
import re
  

nltk.download("stopwords")
nltk.download('wordnet')

class BinomialUtils(SentenceUtils,WordUtils):
    

    def __init__(self, title, content):
        self.title = title
        self.content = content

    def getTitle(self):
        return self.title

    def getContent(self):
        return self.content
    #ok 
    def getBinomialDistribution(self):
        tempLength = 0
        tempHolder = []
        tempLength2 = 0
        generalList = []
        overAllList = []
        sortedDictionary = []
        sortedDictionary2 = []
        sortedDictionary3 = []
        sortedDictionary4 = []
        tempInner = []
        finalInner = []
        senConstruct = []
        senConstruct2 = " "
        senConstruct3 = []
        keyList =[]
        tempDict = []
        finalDict= []
        temp2Dict = []
        final2Dict= []
        temp3Dict = []
        final3Dict= []
        temp4Dict = []
        final4Dict= []
        storageTempSimi = []
        storageTempSimi2 = []
        storageTempSimi3 = []
        inter1 = []
        dividedEq2Result = []
        dividedEq2Result2 = []
        tempBigram = []
        tempSentenceList = []
        equation1Result= []
        tempUniqueWords = []
        eQ2 = []
        equation2Result = []
        eQ3 = []
        tW = []
        tWResult = []
        sW = []
        sWResult = []
        equation3Result = []
        values = []
        frequency = []
        holderSentenceList = []
        tempHolderSentenceList = []
        splitSentence = []
        occur = []
        tempSpecList = []
        finalSpecList = []
        tempSpecList2 = []
        finalSpecList2 = []
        tempFinalSimi = []
        newFinalSimi = []
        compareEqualizer = []
        nonRedundant = []
        redundant = []
        holderFre = []
        finalFre = []
        coOccur = []
        coOccurence = []
        perCoOccur = []
        perCoOccur2 = []
        toBeComparedEqualizer = []
        perCoOccurence = []
        uniCtr = 0
        senCtr = 0
        senCtr2 = 0
        occurCtr = 0
        coopCtr = 0
        coop2Ctr = 0
        coop3Ctr = 0
        tempCtr = 1
        temp2Ctr = 0
        coOp1 = 0
        coOp2 = 0
        coOp3 = 0
        coOp4 = 0
        coOp = 0
        idPass = 0
        coOper = 0 
        lenghtEQ1 = 0
        ctrSimi = 0
        eQ1Len = 0
        tempEQ2 = 0
        sorterCtr = 0
        tempEqualizerlist1 = []
        tempEqualizerlist2 = []
        finalEqualizer = []
        suroUniqueWords = []
        blabla= []
        tempHoldInner = []
        finalHoldInner = []
        tempSimi =[]
        finalSimi = []
        reduCtr = 0
        reduCtr2 = 0
        equalizerSwitch = 0
        c = 0
        cartCtr2 = 0
        w = 0
        x = 0
        y = 0
        z = 0
        
        
    # stop words 
        paraGraph = self.content
        paraGraph2 = sent_tokenize(paraGraph)
        paraGraph = paraGraph.lower()
        # print(paraGraph2)
        stop_words = set(stopwords.words("english"))
        sentenceLister = " ".join([word for word in paraGraph.split() if word not in stop_words])
        noStopWordsList = sentenceLister

    # list of sentence (let S) - status: ok working 

        tempHolder = sent_tokenize(noStopWordsList)
        
        tempLength = len(tempHolder)

        
        # tempHolder = noStopWordsList.split(".")
        # tempHolder.pop() #temporary just this cases basis only need help with Cyrus. remove this and print finalSentenceList[0] you will see extra list with no elements messes with ocurrence checking
        # 
        # for senCtr in range(tempLength):
        #     tempHolderSentenceList.insert(senCtr,tempHolder[senCtr])
        # senCtr = 0
        
        # print(tempHolder)
        

        lemmatizer = WordNetLemmatizer()

        for sentence in tempHolder:
            
            sentence = ''.join([word for word in sentence])
            sentence = ''.join([l for l in re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+://\S+)|^rt|[0-9]|http.+?", "", sentence) if l not in ('!', '.', ',', '/', '\\', ',', ':',
                               ';', '?', '.', '[', ']', '(', ')', '-', '*', '“', '”', '"', '\’', '\‘','\n', '\r', '\t')])
            sentence = sentence.strip()
            sentence = ''.join([lemmatizer.lemmatize(word) for word in sentence])
            holderSentenceList.append(sentence)

        # print(holderSentenceList)
        

        for biCtr in range(tempLength):
            nltk_tokens = nltk.word_tokenize(holderSentenceList[biCtr]) 
            # print(nltk_tokens) 	
            tempBigram.append(list(nltk.bigrams(nltk_tokens)))
            finalEqualizer.append(nltk_tokens) 
   
        generalList.append(tempBigram)
        
        
        # for numCtRR in range(tempLength):
        #     print("this is nltk")
        #     print(finalEqualizer[numCtRR])
            
        for numCtrr in range(tempLength):
            suroUniqueWords.extend(finalEqualizer[numCtrr])
            
        # print(suroUniqueWords)
        
        tempUniqueWords = suroUniqueWords
        tempLength2 = len(tempUniqueWords)
        # print(tempUniqueWords)
        
        
        # print(tempBigram)
        
        
    #list of Uniques (let T) - status: ok Working

        # suroUniqueWords = noStopWordsList.split(" ")
        # # print(suroUniqueWords)
        # suroUniqueWords = list(dict.fromkeys(suroUniqueWords)) # duplicate remover
        # tempLength2 = len(suroUniqueWords)

        # for sentence in suroUniqueWords:
        #     # print(sentence)
        #     sentence = ''.join([word for word in sentence])
        #     sentence = ''.join([l for l in re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+://\S+)|^rt|[0-9]|http.+?", "", sentence) if l not in ('!', '.', ',', '/', '\\', ',', ':',
        #                        ';', '?', '.', '[', ']', '(', ')', '-', '*', '“', '”', '"', '\’', '\‘','\n', '\r', '\t')])
            
        #     sentence = sentence.strip()
            
        #     sentence = ''.join([lemmatizer.lemmatize(word) for word in sentence])
        
        #     tempUniqueWords.append(sentence)

        # print(tempUniqueWords)
        # print(len(tempUniqueWords))
        
    # nested list - status : ok working 
    # this is tokenization
        for senCtr in range(tempLength):
            splitSentence = holderSentenceList[senCtr].split(" ")
            tempSentenceList.insert(senCtr,splitSentence)
        senCtr = 0
        generalList.append(tempSentenceList)

        convertedNumOfSen = int(tempLength)
        convertedNumOfUni = int(tempLength2) 


        # for numCtRR in range(convertedNumOfSen):
        #     print(tempSentenceList[numCtRR])

        overAllList = [
            
            tempUniqueWords,
            generalList
        ]

    #number of times unique word ocurred - status : ok but needs adjusting  
                         
    
        for senCtr2 in range(convertedNumOfSen):
            for word2 in tempUniqueWords:    
                if(word2 in tempSentenceList[senCtr2]):
                    one = 1
                    frequency.append(one)
                else:
                    zero = 0
                    frequency.append(zero)
            values.insert(senCtr2,frequency)
            frequency = []
        generalList.append(values)

        # for numCtRR in range(convertedNumOfSen):
            # print(values[numCtRR])

        # adding ocurrencences for EQ 1  - status: ok working needs adjustments 

        for uniCtr in range(convertedNumOfUni):
            #print(uniCtr , " eto yung for")
            for x in range(convertedNumOfSen):
                num = int(values[x][uniCtr])
                #print(num , "yung number")
                if(num > 0):
                    y += 1   
                if(num < 1):   
                    y += 0         
            occur.append(y)    
            y = 0
        generalList.append(occur)
        tempLength3 = len(occur)
        convertedNumOfOccur = int(tempLength3) 
        #print(generalList[0])

        # print(occur)
        # print(len(occur))

        # Equation 1 - status: ok working / also subject to changes

        for occurCtr in range(convertedNumOfOccur):
            numOccur = float(occur[occurCtr])
            z = numOccur/convertedNumOfSen
            equation1Result.append(z)
        generalList.append(equation1Result)

        # print("this is Equation1")
        # print(equation1Result)
        # print(len(equation1Result))
        lenghtEQ1 = len(equation1Result)
        eQ1Len = int(lenghtEQ1)

        # for numCtRR in range(convertedNumOfSen):
        #     print(tempBigram[numCtRR])

        # co-occur for EQ 2 - status: ok working open to testing

    
        for coop3Ctr in range(convertedNumOfSen):
            for coopCtr in range(convertedNumOfSen):
                tempLength4 = len(tempBigram[coopCtr])
                convertedNumOfBigram = int(tempLength4)
                for coop2Ctr in range(convertedNumOfBigram):
                    if((tempBigram[coopCtr][coop2Ctr][0] and tempBigram[coopCtr][coop2Ctr][1])in tempSentenceList[coop3Ctr]):
                        coOp1 = 1
                        coOccur.append(coOp1)  
                    else:
                        coOp1 = 0
                        coOccur.append(coOp1)        
                coOccurence.append(coOccur)
                coOccur = []
            perCoOccurence.append(coOccurence)
            coOccurence = []
        
        # for numCtRR in range(convertedNumOfSen):
        #     print(perCoOccurence[numCtRR])
        #     print(len(perCoOccurence[numCtRR]))

        # co-occurence per sentence - status : ok open to testing
        
       
        for coOp2 in range(convertedNumOfSen):
            tempLength5 = len(perCoOccurence[temp2Ctr][coOp2])
            for coOp1 in range(tempLength5):
                for coOp3 in range(convertedNumOfSen):
                    numOccur = int(perCoOccurence[coOp3][coOp2][coOp1])
                    if(numOccur > 0):
                        coOper += 1
                    if(numOccur == 0):
                        coOper += 0
                perCoOccur.append(coOper)
                coOper = 0
            perCoOccur2.append(perCoOccur)
            perCoOccur = []
            temp2Ctr +=1
        generalList.append(perCoOccur2)
            
        tempLength6 = len(perCoOccur2)
        convertedNumOfPerOccur = int(tempLength6)

        # for numCtRR in range(convertedNumOfSen):
        #     print(perCoOccur2[numCtRR])
        #     print(len(perCoOccur2[numCtRR]))

        #specifying which list of probability to be used  for each sentence

        for specCtr in range(convertedNumOfSen):
            specLenght = int(len(tempBigram[specCtr]))
            for specCtr2 in range(specLenght):
                holderSpec = tempUniqueWords.index(tempBigram[specCtr][specCtr2][0])
                tempSpecList.append(equation1Result[holderSpec])
            finalSpecList.append(tempSpecList)
            tempSpecList = []
            
        # for numCtRR in range(convertedNumOfSen):
        #     print(finalSpecList[numCtRR]) 
        #     print(len(finalSpecList[numCtRR]))
        
        # specify which number of trials for each sentence
        for specCtr in range(convertedNumOfSen):
            specLenght = int(len(tempBigram[specCtr]))
            for specCtr2 in range(specLenght):
                holderSpec2 = tempUniqueWords.index(tempBigram[specCtr][specCtr2][0])
                tempSpecList2.append(occur[holderSpec2])
            finalSpecList2.append(tempSpecList2)
            tempSpecList2 = []
        
        # for numCtRR in range(convertedNumOfSen):
            # print(finalSpecList2[numCtRR]) 
            # print(len(finalSpecList2[numCtRR]))
        # print(finalSpecList)
           

        # Equation 2 - status: still observing
        for coOp3 in range(convertedNumOfPerOccur):
            tempLength4  = len(perCoOccur2[coOp3])
            # for coopCtr in range(convertedNumOfUni):
            #     coopCtr += 1
            for coOp4 in range(tempLength4):
                eQ2Temp = finalSpecList[coOp3][coOp4]
                eQ2Temp2 = perCoOccur2[coOp3][coOp4]
                # print("this is binomial ")
                # print(convertedNumOfSen)
                # print(eQ2Temp)
                # print(eQ2Temp2)
                convertedNumOftrials = int(finalSpecList2[coOp3][coOp4])
                print("this is trials")
                print(convertedNumOftrials)
                k = list(range(convertedNumOftrials + 1))
                eQ2.append([binompmf(eQ2Temp2,n,eQ2Temp) for n in k])
            equation2Result.append(eQ2)
            eQ2 = []
        generalList.append(equation2Result)

        # for numCtRR in range(convertedNumOfSen):
        #     print("partition")
        #     print(equation2Result[numCtRR])
        # Equation 3 - status: ok working 
        
        for eq3Ctr in range(convertedNumOfSen):
            eQ2lenght = len(equation2Result[eq3Ctr])
            convertedEq2Lenght = int(eQ2lenght)
            for eq3Ctr2 in range(convertedEq2Lenght): 
                eQ3Result = -1 * (np.log2(equation2Result[eq3Ctr][eq3Ctr2]))
                eQ3.append(eQ3Result)
            equation3Result.append(eQ3)
            eQ3 = []
        generalList.append(equation3Result)
        
        # for numCtRR in range(convertedNumOfSen):
        #     print("partion")
        #     print(equation3Result[numCtRR])

        # term weight - status: ok working

        for tWCtr in range(convertedNumOfSen):
            tWLenght = len(equation3Result[tWCtr])
            for tWCtr2 in range(tWLenght):
                tempTw = sum(equation3Result[tWCtr][tWCtr2])
                hue = tempTw
                tW.append(hue)
            tWResult.append(tW)
            tW = []
        generalList.append(tWResult)

        lenghtTW = len(tWResult)
        convertedTwLenght = int(lenghtTW)

        # for numCtRR in range(convertedNumOfSen):
        #     print("partion")
        #     print(tWResult[numCtRR])

        # sentence weight - status: ok working

        for sWCtr in range(convertedNumOfSen):
            tempSw = sum(tWResult[sWCtr])
            sWResult.append(tempSw)
        generalList.append(sWResult)

        print(sWResult)

        print(len(sWResult))

        # convertedNumOfSw = int(len(sWResult))
        
        # sorting sentence weight 
        
        for key in range(convertedNumOfSen):
            keyList.append(key)
            
        #print(keyList)
        for dict2Ctr in range(convertedNumOfSen):
            tempDict.append(keyList[dict2Ctr])
            tempDict.append(sWResult[dict2Ctr])
            finalDict.append(tempDict)
            tempDict = []
            
        # for numCtRR in range(convertedNumOfSen):
            
        #     print(finalDict[numCtRR])
        # print("partion")
            
        # print("dictionary")
        # print(finalDict)
        
        sWResult.sort(reverse=True)
        # print(sWResult)
        
        # sorting sen weight 
    
        for sortCtr in range(convertedNumOfSen):
            remoLenght = int(len(finalDict))
            for sortCtr2 in range(remoLenght):
                if(sWResult[sortCtr] == finalDict[sortCtr2][1]):
                    if (finalDict[sortCtr2] not in sortedDictionary):
                        sortedDictionary.append(finalDict[sortCtr2])
        # print("division")                  
        # for numCtRR in range(convertedNumOfSen):
           
        #     print(sortedDictionary[numCtRR])
                    
        # frequency for innerproduct

        for freCtr in range(convertedNumOfSen):
            for freCtr2 in range(convertedNumOfUni):
                tempFre = tempSentenceList[freCtr].count(tempUniqueWords[freCtr2])
                holderFre.append(tempFre)
            finalFre.append(holderFre)
            holderFre = []

        # for numCtRR in range(convertedNumOfSen):
        #     print("partion")
        #     print(finalFre[numCtRR])
        
        # frequency  with keys
        
        for dict2Ctr2 in range(convertedNumOfSen):
            temp2Dict.append(keyList[dict2Ctr2])
            temp2Dict.append(finalFre[dict2Ctr2])
            final2Dict.append(temp2Dict)
            temp2Dict = []
            
          
        # for numCtRR in range(convertedNumOfSen):
        #     print("partion")
        #     print(final2Dict[numCtRR])
            
        # sentences with keys 
        
        for dict2Ctr3 in range(convertedNumOfSen):
            temp3Dict.append(keyList[dict2Ctr3])
            temp3Dict.append(tempSentenceList[dict2Ctr3])
            final3Dict.append(temp3Dict)
            temp3Dict = []
        
        # for numCtRR in range(convertedNumOfSen):
        #     print("partion")
        #     print(final3Dict[numCtRR])
        
        # original sentence with keys
        
        for dict2Ctr4 in range(convertedNumOfSen):
            temp4Dict.append(keyList[dict2Ctr4])
            temp4Dict.append(paraGraph2[dict2Ctr4])
            final4Dict.append(temp4Dict)
            temp4Dict = []
        
         
        # for numCtRR in range(convertedNumOfSen):
        #     print("partion")
        #     print(final4Dict[numCtRR])
        
        
        # sorting sentence according to sorted sen weight
        for sortCtr in range(convertedNumOfSen):
            remoLenght = int(len(final2Dict))
            for sortCtr2 in range(remoLenght):
                if(sortedDictionary[sortCtr][0] == final3Dict[sortCtr2][0]):
                    sortedDictionary3.append(final3Dict[sortCtr2])
                    
        # for numCtRR in range(convertedNumOfSen):
        #     print("new partion")
        #     print(sortedDictionary3[numCtRR])
        
        # sorting frequency according to sorted sen weight
        
        for sortCtr in range(convertedNumOfSen):
            remoLenght = int(len(final3Dict))
            for sortCtr2 in range(remoLenght):
                if(sortedDictionary[sortCtr][0] == final2Dict[sortCtr2][0]):
                    sortedDictionary2.append(final2Dict[sortCtr2])
        
        # for numCtRR in range(convertedNumOfSen):
        #     print("new partion")
        #     print(sortedDictionary2[numCtRR])
        
        # sorting original sentence according to sen weight 
        
        for sortCtr in range(convertedNumOfSen):
            remoLenght = int(len(final4Dict))
            for sortCtr2 in range(remoLenght):
                if(sortedDictionary[sortCtr][0] == final4Dict[sortCtr2][0]):
                    sortedDictionary4.append(final4Dict[sortCtr2])
                    
        # for numCtRR in range(convertedNumOfSen):
        #     print("new partion")
        #     print(sortedDictionary4[numCtRR])
        
        # print(sortedDictionary4[0][1])
        
        # similarity measure
        
        for simiCtr in range(int(len(sortedDictionary2))):
            for simiCtr2 in range(int(len(sortedDictionary2))):
                bow1 = sortedDictionary2[simiCtr][1]
                bow2 = sortedDictionary2[simiCtr2][1]
                a = np.inner(bow1,bow2)/ (norm(bow1) * norm(bow2))
                tempSimi.append(a)
            finalSimi.append(tempSimi)
            tempSimi = []      
         
        
        # for numCtRR in range(convertedNumOfSen):
        #     print("new partion")
        #     print(finalSimi[numCtRR])   

        # remove its similarity with itself
        
        for remoCtr in range(convertedNumOfSen):
            finalSimi[remoCtr].pop(remoCtr)
            
        # for numCtRR in range(convertedNumOfSen):
        #     print("new partion")
        #     print(finalSimi[numCtRR])   
        
        #assign correct keys 
        
        for assignCtr in range(convertedNumOfSen):
            tempFinalSimi.append(sortedDictionary2[assignCtr][0])
            tempFinalSimi.append(finalSimi[assignCtr])
            newFinalSimi.append(tempFinalSimi)
            tempFinalSimi = []
            
        nonRedundant.append(newFinalSimi[0][0])
        newFinalSimi.pop(0)
        nonRedunLenght = int(len(nonRedundant))
       
        # for numCtRR in range(int(len(newFinalSimi))):
        #     print("new partion")
        #     print(newFinalSimi[numCtRR])   
        
        # print(nonRedundant)
        
        b = 1
        
        # specified checking of similarity
        
        for stoCtr in range(int(len(newFinalSimi))):
            # print("new")
            # print(stoCtr)
            # print(b)
            for stoCtr2 in range(b):
                # print("new iter", " " , stoCtr2)
                storageTempSimi.append(newFinalSimi[stoCtr][1][stoCtr2])
            storageTempSimi2.append(newFinalSimi[stoCtr][0])
            storageTempSimi2.append(storageTempSimi)
            storageTempSimi = []
            storageTempSimi3.append(storageTempSimi2)
            storageTempSimi2 = []
            # print("old")            
            # print(b)
            b+=1
            # print("updated")
            # print(b)
        b = 0
        # print("final b:", " ", b)   
    
        # for numCtRR in range(int(len(storageTempSimi3))):
        #     print("new partion")
        #     print(storageTempSimi3[numCtRR])   
        
        # storing redundant and non redundant sentences key
        
        for reduCtr in range(int(len(storageTempSimi3))):
            reduLenght = int(len(storageTempSimi3[reduCtr][1]))
            for reduCt2 in range(reduLenght):
                nonRedu = storageTempSimi3[reduCtr][1][reduCt2]
                if(nonRedu >= 0.66):
                    idPass += 1
                else:
                    idPass += 0
            # print("this is id paas")
            # print(idPass)
            if(idPass > 0):
                redundant.append(storageTempSimi3[reduCtr][0])
                idPass = 0    
            else:
                nonRedundant.append(storageTempSimi3[reduCtr][0])
                idPass = 0 
                
                
        # print("hello")
        # print(nonRedundant)
        # print(redundant)
        
        # construct new sentence 
        
        for consCtr in range(int(len(nonRedundant))):
            for cons2Ctr in range(int(len(sortedDictionary4))):
                if(nonRedundant[consCtr] == sortedDictionary4[cons2Ctr][0]):
                    print(nonRedundant[consCtr])
                    print(sortedDictionary4[cons2Ctr][0])
                    print(sortedDictionary4[cons2Ctr][1])
                    senConstruct.append(sortedDictionary4[cons2Ctr][1])
        # print(senConstruct)
        senConstruct2 = (' '.join(senConstruct))
        # print(senConstruct2)
        senConstruct3.append(senConstruct2)
        
        return senConstruct3

        
def factorial(n):
    x = 1
    for i in range(1, n+1):
        x *= i
    return x

def combination(n, k):
    return factorial(n)/(factorial(k)*factorial(n-k))

def binompmf(k,n,p):
    # print(k)
    # print(n)
    # print(p)
    if(k > n):
       safe = -1*(n-k)
    else:
        safe = (n-k)
    return combination(n,k)*(p**k)*((1-p)**safe)

# def innerProduct(bow1, bow2):
#     inner_product = 0.0
#     for key in range(len(bow1)):
#         inner_product += bow1[key] * bow2[key]
#     sum1 = 0.0
#     sum2 = 0.0
#     for v in bow1:
#         sum1 += v*v
#     for v in bow2:
#         sum2 += v*v
#     inner_product /= np.sqrt(sum1 * sum2)
#     print(inner_product)
#     return inner_product
    

# def cosineSimilarity(sentence_bow, list_of_sentence_bow):
#     inner_product = innerProduct(sentence_bow, list_of_sentence_bow)
#     if inner_product >= 0.66:
#         return True
#     return False

        











        


                    






        

    
    









       

        

 
        

        






   