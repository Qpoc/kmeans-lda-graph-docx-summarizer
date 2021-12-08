import numpy as np

def randCent(dataSet,K):
    n= np.shape(dataSet)[1]
    centroids=np.mat(np.zeros((K,n)))
    for j in range(n):
        minValue=min(dataSet[:,j])
        maxValue=max(dataSet[:,j])
        rangeValues=float(maxValue-minValue)
        #Make sure centroids stay within the range of data
        centroids[:,j]=minValue+rangeValues*np.random.rand(K,1)
    # np.matrix
    return centroids

# euclidean distance measure
def distanceMeasure(vecOne, vecTwo):
    return np.sqrt(np.sum(np.power(vecOne-vecTwo,2)))

# K means clustering method
def kMeans(dataSet,K,distMethods=distanceMeasure,createCent=randCent):
    m=np.shape(dataSet)[0]
    clusterAssment=np.mat(np.zeros((m,2)))
    # np.matrix
    centroids=createCent(dataSet,K)
    clusterChanged=True
    
    while clusterChanged:
        clusterChanged=False
        for i in range(m):
            minDist=np.inf; minIndex=-2
            for j in range(K):
                distJI=distMethods(centroids[j,:],dataSet[i,:])
                if distJI<minDist:
                    minDist=distJI;minIndex=j
            if clusterAssment[i,0] != minIndex:
                clusterChanged=True
            clusterAssment[i,:]=minIndex,minDist**2
        #update all the centroids by taking the np.mean value of relevant data
        for cent in range(K):
            ptsInClust=dataSet[np.nonzero(clusterAssment[:,0].A == cent)[0]]
            centroids[cent,:]=np.mean(ptsInClust,axis=0)
    # (np.matrix, np.matrix)
    return centroids,clusterAssment

#bisecting K-means method
def bisectingKMeans(dataSet,K,numIterations):
    m,n=dataSet.shape
    clusterInformation = np.mat(np.zeros((m,2)))
    centroidList=[]
    minSSE = np.inf
    
    #At the first place, regard the whole dataset as a cluster and find the best clusters
    for i in range(numIterations):
        # (np.matrix, np.matrix)
        centroid,clusterAssment=kMeans(dataSet, 2)
        SSE=np.sum(clusterAssment,axis=0)[0,1]
        if SSE<minSSE:
            minSSE=SSE
            tempCentroid=centroid
            tempCluster=clusterAssment
    centroidList.append(tempCentroid[0].tolist()[0])
    centroidList.append(tempCentroid[1].tolist()[0])
    clusterInformation=tempCluster
    minSSE=np.inf 
    
    while len(centroidList)<K:
        maxIndex=-2
        maxSSE=-1
        #Choose the cluster with Maximum SSE to split
        for j in range(len(centroidList)):
            SSE=np.sum(clusterInformation[np.nonzero(clusterInformation[:,0]==j)[0]])
            if SSE>maxSSE:
                maxIndex=j
                maxSSE=SSE
                
        #Choose the clusters with minimum total SSE to store into the centroidList
        for k in range(numIterations):
            pointsInCluster=dataSet[np.nonzero(clusterInformation[:,0]==maxIndex)[0]]
            # (np.matrix, np.matrix)
            centroid,clusterAssment=kMeans(pointsInCluster, 2)
            SSE=np.sum(clusterAssment[:,1],axis=0)
            if SSE<minSSE:
                minSSE=SSE
                tempCentroid=centroid.copy()
                tempCluster=clusterAssment.copy()
        #Update the index
        tempCluster[np.nonzero(tempCluster[:,0]==1)[0],0]=len(centroidList)
        tempCluster[np.nonzero(tempCluster[:,0]==0)[0],0]=maxIndex
        
        #update the information of index and SSE
        clusterInformation[np.nonzero(clusterInformation[:,0]==maxIndex)[0],:]=tempCluster
        #update the centrolist
        centroidList[maxIndex]=tempCentroid[0].tolist()[0]
        centroidList.append(tempCentroid[1].tolist()[0])
    # (List[List[float]], np.matrix)
    return clusterInformation
