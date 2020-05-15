
import numpy as np
import math 

'''
信息熵计算
H=−∑ni=1p(xi)log2p(xi)
'''
def calcShannonEnt(dataset) : 
    numEntries = len(dataset)
    labelCount = {}
    for featVec in range(numEntries):
        currentLabel = featVec[-1]
        if labelCount[currentLabel] not in labelCount.keys():
            labelCount[currentLabel] = 0 
        labelCount[currentLabel]+= 1 
    shannonEnt = 0.0
    for keys in labelCount.keys() :
        prob = labelCount[keys]/numEntries
        shannonEnt -= prob * math.log(prob,2)
    return shannonEnt