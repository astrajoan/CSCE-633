# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 09:49:10 2020

@author: Ziqi Zhao
"""

import csv
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import math


def importFile(filename):
    
    datafile = csv.reader(open(filename,'r'))
    datafile = list(datafile)
    
    del datafile[0]
    
    num = []
    cat = []
    label = []
    
    for idx, line in enumerate(datafile):
        del line[0]
        num.append([float(ele) for i, ele in enumerate(line) if (i!=6 and i!=7)])
        cat.append([ele for i,ele in enumerate(line) if i==6])
        label.append([float(ele) for i,ele in enumerate(line) if i==7])
    
    return np.array(num),np.array(cat),np.array(label).flatten()


class pearson(object):
    
    def __init__(self,num,y):
        self.num = num
        self.y = y
        
    def pearsonCalcXy(self):
        for i in range(len(self.num[0])):
            corr, _ = pearsonr(self.num[:,i], self.y)
            print('Pearson correlation for {}: {}'.format(i+1,corr))
            
    def pearsonPlotXy(self,savefig):
        ncol = 3
        nrow = 2
        titles = ['stamina - combat point','attack value - combat point',
                  'defense value - combat point',
                  'capture rate - combat point','flee rate - combat point',
                  'spawn chance - combat point']
        colors = ['k','b','g','r','orange','y']
        plt.figure(figsize=(6*ncol,6*nrow))
        for i in range(len(self.num[0])):
            plt.subplot(nrow,ncol,i+1)
            plt.scatter(self.num[:,i],self.y,marker='.',color=colors[i])
            plt.title(titles[i])
            plt.xlabel('Feature value')
            plt.ylabel('y (combat point)')
            plt.grid(axis='both')
        
        if savefig == True:
            plt.savefig('D:/Github/CSCE-633/Homeworks/Homework2/Numerical feature - y.png',dpi=120)
    
    def pearsonCalcXX(self):
        for i in range(len(self.num[0])-1):
            for j in range(i+1,len(self.num[0])):
                corr, _ = pearsonr(self.num[:,i], self.num[:,j])
                print('Pearson correlation for {} and {} : {}'.format(i+1,j+1,corr))
            print('\n')
    
    def pearsonPlotXX(self,savefig):
        ncol = 3
        nrow = 5
        n = 0
        
        titles = [['stamina - attack value','stamina - defense value','stamina - capture rate',
                   'stamina - flee rate','stamina - spawn chance'],
                  ['attack value - defense value','attack value - capture rate',
                   'attack value - flee rate','attack value - spawn chance'],
                  ['defense value - capture rate','defense value - flee rate',
                   'defense value - spawn chance'],
                  ['capture rate - flee rate','capture rate - spawn chance'],
                  ['flee rate - spawn chance']]
        colors = ['k','b','g','r','orange','y']
        plt.figure(figsize=(6*ncol,6*nrow))
        for i in range(len(self.num[0])-1):
            for j in range(i+1,len(self.num[0])):
                plt.subplot(nrow,ncol,n+1)
                plt.scatter(self.num[:,i],self.num[:,j],marker='.',color=colors[i])
                plt.title(titles[i][j-i-1])
                plt.xlabel('Feature1')
                plt.ylabel('Feature2')
                plt.grid(axis='both')
                n = n+1
            
        if savefig == True:
            plt.savefig('D:/Github/CSCE-633/Homeworks/Homework2/Numerical feature - Numerical feature.png',dpi=120)


def oneHotEncoding(cat):
    x = [[0]*15 for i in range(len(cat))]
    catMap = {'Grass':0,'Fire':1,'Water':2,'Bug':3,'Normal':4,'Poison':5,
              'Electric':6,'Ground':7,'Fairy':8,'Fighting':9,'Psychic':10,
              'Rock':11,'Ghost':12,'Ice':13,
              'Dragon':14}
    
    for i in range(len(cat)):
        x[i][catMap[cat.flatten()[i]]] = 1
    
    return np.array(x)


def normalizeParams(feats):
    mean = []
    std = []
    
    for i in range(len(feats[0])):
        mean.append(np.mean(feats[:,i]))
        std.append(np.std(feats[:,i]))
    
    return mean,std


def normalize(feats,mean,std):
    
    feats = (feats - mean)/std

    return feats


def splitData(cvX,cvY,idx,fold):
    trainX = []
    trainY = []
    valX = []
    valY = []
    n = math.floor(cvX.shape[0]/5)
    
    valX = cvX[n*fold:n*(fold+1)+1,:] # Validation set contains 30 samples
    valY = cvY[n*fold:n*(fold+1)+1]
    
    trainX = np.concatenate((cvX[0:n*fold,:],cvX[n*(fold+1)+1:,:]),axis=0)
    trainY = np.concatenate((cvY[0:n*fold],cvY[n*(fold+1)+1:]),axis=0)
    
    return trainX,trainY,valX,valY
    
    
class linearRegression(object):
    
    def __init__(self,w):
        self.w = w
        #print(w)
        
    def gradient(self,trainX,trainY):
        grads = np.matmul(np.matmul(trainX.transpose(),trainX),self.w) - np.matmul(trainX.transpose(),trainY)
        
        return grads
    
    def train(self,trainX,trainY,alpha,iteration):
        err = []
        
        for i in range(iteration):
            pred = self.predict(trainX)
            err.append(self.errorRate(pred,trainY))
            
            grad = self.gradient(trainX,trainY)
            self.w = self.w - alpha * grad
            if i % 10000 == 0:
                print(self.w)
            
        return err
            
    def predict(self,X):
        return np.matmul(X,self.w)
    
    def errorRate(self,pred,y):
        return np.sqrt(sum(np.square(np.subtract(pred,y))))
    
    def printw(self):
        print('The w calculated using gradient descent is {}'.format(self.w))
    
        
            



''' 
    Import Data 
'''
numX,catX,Y = importFile('D:/Github/CSCE-633/Homeworks/Homework2/hw2_data.csv')

'''
    Pearson correlation
'''
#pear = pearson(numX,Y)
#pear.pearsonCalcXy()
#pear.pearsonPlotXy(True)
#pear.pearsonCalcXX()
#pear.pearsonPlotXX(True)

'''
    One hot encoding
'''
catXCode = oneHotEncoding(catX)

X = np.concatenate((numX,catXCode),axis=1)
X = np.concatenate(([[1] for i in range(X.shape[0])],X),axis=1) # Bias term 1

'''
    Use highly correlated features
'''
# =============================================================================
# #X = np.vstack((numX[:,0],numX[:,1])).transpose()
# X = numX[:,0:1]
# X = np.concatenate(([[1] for i in range(X.shape[0])],X),axis=1)
# =============================================================================

'''
    Logistic regression assign labels
'''
yMean = np.mean(Y)
Y = [1 if ele > yMean else 0 for i,ele in enumerate(Y)]

'''
    Normalize data
'''
# =============================================================================
# trainMean, trainStd = normalizeParams(X)
# X = normalize(X,trainMean,trainStd)
# =============================================================================

'''
    Split samples into 5 equal parts
'''
idx = np.random.permutation(X.shape[0]) # Randomly shuffle data index
cvX = np.array([X[idx[i]] for i in range(len(idx))]) # Shuffled X
cvY = np.array([Y[idx[i]] for i in range(len(idx))])

'''
    Linear regression
'''
err = []
errReg = []
wReg = []

lambdas = [1.9e-2, 1e-1, 10, 1e-3, 1.9e-3]

for i in range(5):
    trainX,trainY,valX,valY = splitData(cvX,cvY,idx,i)
    
    Xinv = np.linalg.pinv(np.dot(trainX.transpose(),trainX))
    Xy = np.matmul(trainX.transpose(),trainY)
    w = np.matmul(Xinv,Xy)
    
    wReg = []
    for j in range(5):
        regXinv = np.linalg.pinv(np.matmul(trainX.transpose(),trainX)+
                              lambdas[j]*np.identity(trainX.shape[1]))
        wReg.append(np.matmul(regXinv,Xy))
    
    lr = linearRegression(w)
    err.append(lr.errorRate(lr.predict(valX),valY))
    
    errRegtemp = []
    for j in range(5):
        lrReg = linearRegression(wReg[j])
        errRegtemp.append(lrReg.errorRate(lrReg.predict(valX),valY))
    
    errReg.append(errRegtemp)

print(err)
print('Without regularization: {}'.format(np.array(err).mean()))
print('Regularization 0: {}'.format(np.array(errReg)[:,0].mean()))
print('Regularization 1: {}'.format(np.array(errReg)[:,1].mean()))
print('Regularization 2: {}'.format(np.array(errReg)[:,2].mean()))
print('Regularization 3: {}'.format(np.array(errReg)[:,3].mean()))
print('Regularization 4: {}'.format(np.array(errReg)[:,4].mean()))

'''
    Logistic regression
'''
lambdas = [1e-3,1e-2,1e-1,1,1e1]
#lambdas = np.linspace(0.1,10,1000)
accTotal = []

# Without regularization
"""
    Use the first 80% of randomly shuffled data as training set, and the other
    20% as test set
"""
trainX,trainY,testX,testY = splitData(cvX,cvY,idx,0)
logr = LogisticRegression(penalty='none',solver='lbfgs')
logr.fit(trainX,trainY)

pred = logr.predict(testX)
acc = np.array([1 if pred[i]==testY[i] else 0 for i in range(testX.shape[0])])
#accTotal.append(acc.mean())

print('No regularization: {}\n\n'.format(acc.mean()))

bestLambda = 0
accPrev = 0
# With regularization
for idx,lam in enumerate(lambdas):
    accReg = []
    for i in range(5):
        trainX0,trainY0,valX,valY = splitData(trainX,trainY,idx,i)
        logrReg = LogisticRegression(penalty='l2',solver='lbfgs',C=lam)
        logrReg.fit(trainX0,trainY0)
        
        pred = logrReg.predict(valX)
        acc = np.array([1 if pred[i]==valY[i] else 0 for i in range(valX.shape[0])])
        accReg.append(acc.mean())
        
    if np.array(accReg).mean()>=accPrev:
        accPrev = np.array(accReg).mean()
        bestLambda = lam
        
    print('Regularization with {}th lambda: {}\n\n'.format(idx,np.array(accReg).mean()))

logrTotal = LogisticRegression(penalty='l2',solver='lbfgs',C=bestLambda)
logrTotal.fit(trainX,trainY)

pred = logrTotal.predict(testX)
accB = np.array([1 if pred[i]==testY[i] else 0 for i in range(testX.shape[0])])

print('The best hyperparameter lambda is {}\n\n'.format(1/bestLambda))
print('Accuracy using best lambda {}\n\n'.format(accB.mean()))

