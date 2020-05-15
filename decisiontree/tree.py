
import numpy as np
import math 

'''
信息熵计算
H=−∑p(xi)log2p(xi)
'''
def calcShannonEnt(dataset) : 
    numEntries = len(dataset)
    labelCount = {}
    for featVec in dataset :     
        currentLabel = featVec[-1]
        if currentLabel not in labelCount.keys():
            labelCount[currentLabel] = 0 
        labelCount[currentLabel]+= 1 
    shannonEnt = 0.0
    for keys in labelCount.keys() :
        prob = labelCount[keys]/numEntries
        shannonEnt -= prob * math.log(prob,2)
    return shannonEnt
def splitDataset(dataSet,axis,value):
    retDataSet = []
    for reduceFeatVec in dataSet : 
        if reduceFeatVec[axis]==value : 
            reduceFeatVec = reduceFeatVec[:axis].extend(reduceFeatVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet



def createDataSet():
    dataSet = [[1,1,'yes'],
              [1,1,'yes'],
              [1,0,'no'],
              [0,1,'no'],
              [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels 