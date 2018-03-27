# -*- coding: utf-8 -*-
from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX,dataSet,labels,k):                  #inX是你要输入的要分类的“坐标”，dataSet是上面createDataSet的array，\
    # 就是已经有的，分类过的坐标，label是相应分类的标签，k是KNN，k近邻里面的k
    dataSetSize=dataSet.shape[0]                     #dataSetSize是dataSet的行数，用上面的举例就是4行
    diffMat=tile(inX,(dataSetSize,1))-dataSet         #前面用tile，把一行inX变成4行一模一样的（tile有重复的功能，dataSetSize\
    # 是重复4遍，后面的1保证重复完了是4行，而不是一行里有四个一样的），然后再减去dataSet，是为了求两点的距离，先要坐标相减，这个\
    # 就是坐标相减
    sqDiffMat=diffMat**2                              #上一行得到了坐标相减，然后这里要(x1-x2)^2，要求乘方
    sqDistances=sqDiffMat.sum(axis=1)                 #axis=1是列相加，，这样得到了(x1-x2)^2+(y1-y2)^2
    distances=sqDistances**0.5                        #开根号，这个之后才是距离
    sortedDistIndicies=distances.argsort()            #argsort是排序，将元素按照由小到大的顺序返回下标，比如([3,1,2]),它返\
    # 回的就是([1,2,0])
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1            #get是取字典里的元素，如果之前这个voteIlabel是有\
        # 的，那么就返回字典里这个voteIlabel里的值，如果没有就返回0（后面写的），这行代码的意思就是算离目标点距离最近的k个点的\
        # 类别，这个点是哪个类别哪个类别就加1
    soredClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)  #key=operator.itemgetter(1)\
    # 的意思是按照字典里的第一个排序，{A:1,B:2},要按照第1个（AB是第0个），即‘1’‘2’排序。reverse=True是降序排序
    return soredClassCount[0][0]             #返回类别最多的类别

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3)) #创建行3列的0矩阵
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip() #截取掉所有回车字符
        listFromLine = line.split('\t') #将所得的整行数据切割成一个元素列表。
        returnMat[index,:] = listFromLine[0:3]#选取前三个元素，将他们存储到特征矩阵中
        classLabelVector.append(int(listFromLine[-1]))#将最后一列存储进去
        index +=1
    return returnMat,classLabelVector


def autoNorm(dataSet):#归一化数值 利用：newValue = (oldValue-min)/(max-min)
    minVals = dataSet.min(0) #不同行比较得出最小数
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet)) #建立与样本相同维度的0矩阵
    m = dataSet.shape[0]  #将样本第一维度数赋值给m
    normDataSet = dataSet -tile(minVals,(m,1))#复制成与输入矩阵同样大小的矩阵
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

def datingClassTest(hoRatio):
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print "the classifier came back with: %d, the real answer is: %d" %(classifierResult, datingLabels[i])
        if(classifierResult != datingLabels[i]):    errorCount += 1.0
    print "the total error rate is: %f" %(errorCount/float(numTestVecs))

def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(raw_input("percentage of time spend playing video games?"))
    ffMiles = float(raw_input("frequent filer miles earned per year?"))
    iceCream = float(raw_input("liters of icream consumed per year?"))
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print "You will probably like this person: ",resultList[classifierResult - 1]