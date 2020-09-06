# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 15:03:17 2020

@author: yvonn
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import operator as op

trainSet = 'iris_train.csv'
validationSet = 'iris_dev.csv'
testSet = 'iris_test.csv'

class dataPoint(object):
    
    def __init__(self,feats):
        self.sl = feats['sl']
        self.sw = feats['sw']
        self.pl = feats['pl']
        self.pw = feats['pw']
        self.label = feats['label']
        
    def printFeature(self):
        print("Sepal length: {}\nSepal width: {}\nPetal length: {}\nPetal width: {}\nLabel: {}"
              .format(self.sl,self.sw,self.pl,self.pw,self.label))


def importFile(filename):
    
    datafile = csv.reader(open(filename,'r'))
    dataset = []
    labelMap = {'Iris-setosa':-1,'Iris-versicolor':0,'Iris-virginica':1}
    
    for idx,line in enumerate(datafile):
        if idx == 0:
            continue
        sl,sw,pl,pw,label = list(np.float_(line[:-1])) + [labelMap[line[-1]]]
        feats = {'sl':sl,'sw':sw,'pl':pl,'pw':pw,'label':label}
        dataset.append(dataPoint(feats))
        
    return dataset


def classCount(dataset):
    count = []
    
    for data in dataset:
        if data.label == -1:
            count.append(-1)
        elif data.label == 0:
            count.append(0)
        else:
            count.append(1)
            
    setosa = sum(i==-1 for i in count)
    versicolor = sum(i==0 for i in count)
    virginica = sum(i==1 for i in count)
    
    print("Setosa:{}\nVersicolor:{}\nVirginica:{}".format(setosa,versicolor,virginica))
    
    
def plotHist(dataset,save):
    sl = []
    sw = []
    pl = []
    pw = []
    feats = []
    titles = ['Sepal length','Sepal width','Petal length','Petal width']
    
    for data in dataset:
        sl.append(data.sl)
        sw.append(data.sw)
        pl.append(data.pl)
        pw.append(data.pw)
    feats.append(sl)
    feats.append(sw)
    feats.append(pl)
    feats.append(pw)
    
    plt.figure(figsize=(6*4,6))
    for i in range(4):
        plt.subplot(1,4,i+1)
        plt.hist(feats[i],bins=10,fill=False,linewidth=3)
        plt.ylim((0,30))
        plt.grid(axis='both')
        plt.title(titles[i])

    if save == True:
        plt.savefig('Histogram.png',dpi=300)
        
    
def plotScatter(dataset,save):
    setosa = []
    versicolor = []
    virginica = []
    feats = ['Sepal length','Sepal width','Petal length','Petal width']
    
    for data in dataset:
        if data.label == -1:
            setosa.append([data.sl,data.sw,data.pl,data.pw])
        elif data.label == 0:
            versicolor.append([data.sl,data.sw,data.pl,data.pw])
        else:
            virginica.append([data.sl,data.sw,data.pl,data.pw])
            
    nrow = 2
    ncol = 3
    plt.figure(figsize=(ncol*6,nrow*6))
    n = 1
    for i in range(ncol):
        for j in range(i+1,ncol+1):
            plt.subplot(nrow,ncol,n)
            plt.scatter([np.array(setosa)[:,i]],[np.array(setosa)[:,j]],marker='.',color='blue',label='Setosa')
            plt.scatter([np.array(versicolor)[:,i]],[np.array(versicolor)[:,j]],marker='.',color='red',label='Versicolor')
            plt.scatter([np.array(virginica)[:,i]],[np.array(virginica)[:,j]],marker='.',color='green',label='Virginica')
            plt.xlabel(feats[i])
            plt.ylabel(feats[j])
            plt.grid(axis='both')
            plt.legend(loc=1)
            
            n = n + 1
    
    if save == True:
        plt.savefig('Scatter plot.png',dpi=300)
        

def normalize(dataset):
    sl = [data.sl for data in dataset]
    sw = [data.sw for data in dataset]
    pl = [data.pl for data in dataset]
    pw = [data.pw for data in dataset]
    
    slMean = np.mean(sl)
    swMean = np.mean(sw)
    plMean = np.mean(pl)
    pwMean = np.mean(pw)
    
    #print("{},{},{},{}".format(slMean,swMean,plMean,pwMean))
    
    slStd = np.std(sl)
    swStd = np.std(sw)
    plStd = np.std(pl)
    pwStd = np.std(pw)
    
    #print("{},{},{},{}".format(slStd,swStd,plStd,pwStd))
    
    for i in range(len(dataset)):
        dataset[i].sl = (dataset[i].sl-slMean)/slStd
        dataset[i].sw = (dataset[i].sw-swMean)/swStd
        dataset[i].pl = (dataset[i].pl-plMean)/plStd
        dataset[i].pw = (dataset[i].pw-pwMean)/pwStd
    
    return dataset        


class kNN(object):
    
    def __init__(self,K):
        self.K = K
        
    def calcEucDist(self,testset,dataset):
        distances = []
        
        for test in testset:
            testDistance = []
            testSample = np.array([test.sl,test.sw,test.pl,test.pw])
            for data in dataset:
                dataSample = np.array([data.sl,data.sw,data.pl,data.pw])
                testDistance.append(np.linalg.norm(testSample-dataSample))
            distances.append(testDistance)
        
        return distances
    
    def calcL1Norm(self,testset,dataset):
        distances = []
        
        for test in testset:
            testDistance = []
            testSample = np.array([test.sl,test.sw,test.pl,test.pw])
            for data in dataset:
                dataSample = np.array([data.sl,data.sw,data.pl,data.pw])
                testDistance.append(sum(abs(testSample-dataSample)))
            distances.append(testDistance)
        
        return distances
    
    def calcL0Norm(self,testset,dataset):
        distances = []
        
        for test in testset:
            testDistance = []
            testSample = np.array([test.sl,test.sw,test.pl,test.pw])
            for data in dataset:
                dataSample = np.array([data.sl,data.sw,data.pl,data.pw])
                L0 = testSample-dataSample
                L0 = [1 if i == 0 else 0 for i in L0]
                testDistance.append(sum(L0))
            distances.append(testDistance)
        
        return distances
    
    def calcCos(self,testset,dataset):
        distances = []
        
        for test in testset:
            testDistance = []
            testSample = np.array([test.sl,test.sw,test.pl,test.pw])
            for data in dataset:
                dataSample = np.array([data.sl,data.sw,data.pl,data.pw])
                testDistance.append(testSample.dot(dataSample)/(np.linalg.norm(testSample)*np.linalg.norm(dataSample)))
            distances.append(testDistance)
        
        return distances
    
    def findKNN(self,distances):
        idx = []
        
        for testDistance in distances:
            testIdx = np.argpartition(testDistance,self.K)[:self.K]
            idx.append(testIdx)
        
        return idx
    
    def predictClass(self,dataset,idxNN):
        testKNN = []
        testClasses = []
        labels = [-1,0,1]
        
        for idx in idxNN:
            testKNN = np.array(dataset)[idx]
            votes = [0] * 3
            for nn in testKNN:
                if nn.label == -1:
                    votes[0] = votes[0] + 1
                elif nn.label == 0:
                    votes[1] = votes[1] + 1
                else:
                    votes[2] = votes[2] + 1
            #print("{},{},{}".format(setosa,versicolor,virginica))
            majorIdx, major = max(enumerate(votes), key = op.itemgetter(1))
            testClasses.append(labels[majorIdx])
            
        return testClasses
                


'''============================== Import Data ================================='''

datasetTrain = importFile(trainSet)

datasetValidation = importFile(validationSet)

datasetTest = importFile(testSet)

'''========================== Data Visualization =============================='''

#classCount(datasetTrain)

#plotHist(datasetTrain,False)

#plotScatter(datasetTrain,True)

'''============================== Validation Set =============================='''

Acc = []

# =============================================================================
# kNNClassfier = kNN(7)
#     
# datasetTrain = normalize(datasetTrain)
# 
# datasetValidation = normalize(datasetValidation)
# 
# distance = kNNClassfier.calcEucDist(datasetValidation,datasetTrain)
#         
# idxNN = kNNClassfier.findKNN(distance)
# 
# classPredictVal = kNNClassfier.predictClass(datasetTrain,idxNN)
# 
# print(classPredictVal)
# 
# classVal = [val.label for val in datasetValidation]
# 
# print(classVal)
# 
# diff = [1 if i == j else 0 for i,j in zip(classVal,classPredictVal)]
# 
# print(sum(diff)/len(diff))
# =============================================================================

datasetTrain = normalize(datasetTrain) # Need to normalize the data

datasetValidation = normalize(datasetValidation)

for i in range(1,90,2):
    kNNClassfier = kNN(i)

    distance = kNNClassfier.calcEucDist(datasetValidation,datasetTrain)
        
    idxNN = kNNClassfier.findKNN(distance)
    
    classPredictVal = kNNClassfier.predictClass(datasetTrain,idxNN)
    
    #print(classPredictVal)
    
    classVal = [val.label for val in datasetValidation]
    
    #print(classVal)
    
    diff = [1 if i == j else 0 for i,j in zip(classVal,classPredictVal)]
    
    #print(sum(diff)/len(diff))
    
    Acc.append(sum(diff)/len(diff))
    
print(Acc)

plt.figure(figsize=(6,4.6*2))
#plt.subplots_adjust(bottom=0.05,top=0.9)
plt.subplot(2,1,1)
plt.plot(range(1,20,2),Acc[:10],linestyle='dashed',marker='o',color='black',markerfacecolor='red')
plt.xlim((0,20))
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.grid(axis='both')
plt.title('K Nearest Neighbor Validation Accuracy')
plt.subplot(2,1,2)
plt.plot(range(1,90,2),Acc,linestyle='dashed',marker='o',color='black',markerfacecolor='red')
plt.xlim((-1,91))
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.grid(axis='both')
plt.title('K Nearest Neighbor Validation Accuracy')
plt.savefig('Validation Accuracy.png',dpi=100)




'''================================== Testset ==============================='''

# =============================================================================
# AccTest = []
# 
# datasetTest = normalize(datasetTest)
# 
# for i in range(1,90,2):
#     kNNClassfier = kNN(i)
# 
#     distance = kNNClassfier.calcEucDist(datasetTest,datasetTrain)
#         
#     idxNN = kNNClassfier.findKNN(distance)
#     
#     classPredictTest = kNNClassfier.predictClass(datasetTrain,idxNN)
#     
#     #print(classPredictTest)
#     
#     classTest = [test.label for test in datasetTest]
#     
#     #print(classTest)
#     
#     diff = [1 if i == j else 0 for i,j in zip(classTest,classPredictTest)]
#     
#     #print(sum(diff)/len(diff))
#     
#     AccTest.append(sum(diff)/len(diff))
# 
# 
# plt.figure(figsize=(6,4))
# plt.plot(range(1,90,2),AccTest,linestyle='dashed',marker='o',color='black',markerfacecolor='red')
# plt.xlim((-1,91))
# plt.xlabel('Iteration')
# plt.ylabel('Accuracy')
# plt.grid(axis='both')
# plt.title('K Nearest Neighbor Test Accuracy')
# 
# =============================================================================
