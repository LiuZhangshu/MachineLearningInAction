
import numpy as np
import math 

'''
信息熵计算
H=−∑p(xi)log2p(xi)
step1 遍历统计标签个数
step2 计算概率
step3 计算信息熵
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
'''
数据集划分
step1 判断feature是否符合条件
step2 筛选出符合条件的样本，去掉判断特征
step3 返回子矩阵
'''
def splitDataset(dataSet,axis,value):
    retDataSet = []
    for FeatVec in dataSet : 
        if FeatVec[axis]==value : 
            reduceFeatVec = FeatVec[:axis]
            reduceFeatVec.extend(FeatVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet

'''
选择最好的数据集划分方式
step1 计算原始交叉熵
step2 遍历特征
    step3 取特征的枚举值，遍历
        step4 通过特征和枚举值进行数据集划分，得到子集
        step5 计算子集的信息熵增益
    step6 计算最信息增益，得到最佳划分点
'''
def chooseBestFeatureToSplit(dataSet):
    baseEntropy = calcShannonEnt(dataSet)
    numFeature = len(dataSet[0]) -1 
    bestInfoGain = 0.0 
    newEntropy = 0.0
    bestFeature = -1 
    for i in range(numFeature) :
        featList = [example[i] for example in dataSet]
        featList = set(featList)
        for value in featList : 
            subDataSet = splitDataset(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain>bestInfoGain):
            bestInfoGain=infoGain
            bestFeature = i 
    return bestFeature
'''
投票
'''    
def majorityCnt(classList):
    classCnt = {}
    for vote in classList : 
        if vote not in classCnt.keys() :
            classCnt[vote] = 0 
            classCnt[vote] +=1
        
    sortedClass = sorted(classCnt.items(),key=lambda item:item[1],reverse=True)
    return sortedClass[0][0]

def createDataSet():
    dataSet = [[1,1,'yes'],
              [1,1,'yes'],
              [1,0,'no'],
              [0,1,'no'],
              [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels 