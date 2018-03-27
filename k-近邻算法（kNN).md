# k-近邻算法（kNN)

优点：精度高、对异常值不敏感、无数据输入假定。

缺点：计算复杂度高、空间复杂度高。

适用数据范围：数值型和标称型。 

工作原理：存在一个样本数据集合，也称作训练样本集，并且样本集中每个数据都有标签，即知道样本集中每一数据与所属分类的对应关系。输入没有标签的新数据后，将新数据的每个特征与样本集中数据对应的特征进行比较，然后算法提取样本集中特征最相似数据的分类标签。一般来说，我们只选择样本数据集中前k个最相似数据，这就是k-近邻算法k的出处，通常k是不大于20的整数。最后，选取k个最相似数据中出现次数最多的分类，作为新数据的分类。

```python
# -*- coding: utf-8 -*-
from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX,dataSet,labels,k):                  #inX是你要输入的要分类的“坐标”，dataSet是上面createDataSet的array，就是已经有的，分类过的坐标，label是相应分类的标签，k是KNN，k近邻里面的k
    dataSetSize=dataSet.shape[0]                     #dataSetSize是sataSet的行数，用上面的举例就是4行
    diffMat=tile(inX,(dataSetSize,1))-dataSet         #前面用tile，把一行inX变成4行一模一样的（tile有重复的功能，dataSetSize是重复4遍，后面的1保证重复完了是4行，而不是一行里有四个一样的），然后再减去dataSet，是为了求两点的距离，先要坐标相减，这个就是坐标相减
    sqDiffMat=diffMat**2                              #上一行得到了坐标相减，然后这里要(x1-x2)^2，要求乘方
    sqDistances=sqDiffMat.sum(axis=1)                 #axis=1是列相加，，这样得到了(x1-x2)^2+(y1-y2)^2
    distances=sqDistances**0.5                        #开根号，这个之后才是距离
    sortedDistIndicies=distances.argsort()            #argsort是排序，将元素按照由小到大的顺序返回下标，比如([3,1,2]),它返回的就是([1,2,0])
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1            #get是取字典里的元素，如果之前这个voteIlabel是有的，那么就返回字典里这个voteIlabel里的值，如果没有就返回0（后面写的），这行代码的意思就是算离目标点距离最近的k个点的类别，这个点是哪个类别哪个类别就加1
    soredClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)         #key=operator.itemgetter(1)的意思是按照字典里的第一个排序，{A:1,B:2},要按照第1个（AB是第0个），即‘1’‘2’排序。reverse=True是降序排序
    return soredClassCount[0][0]             #返回类别最多的类别
```

