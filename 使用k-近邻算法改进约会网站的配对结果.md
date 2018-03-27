# 使用k-近邻算法改进约会网站的配对结果

## 准备数据：从文本文件中解析数据

将文本记录转换为NumPy的解析程序

```python
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
```

在控制台中输入如下命令：

```python
reload(kNN)
<module 'kNN' from 'C:\Users\wangz\Desktop\ML_study\kNN\kNN.py'>
datingDataMat,datingLabels = kNN.file2matrix('datingTestSet2.txt')
datingDataMat
array([[4.0920000e+04, 8.3269760e+00, 9.5395200e-01],
       [1.4488000e+04, 7.1534690e+00, 1.6739040e+00],
       [2.6052000e+04, 1.4418710e+00, 8.0512400e-01],
       ...,
       [2.6575000e+04, 1.0650102e+01, 8.6662700e-01],
       [4.8111000e+04, 9.1345280e+00, 7.2804500e-01],
       [4.3757000e+04, 7.8826010e+00, 1.3324460e+00]])
datingLabels[0:20]
[3, 2, 1, 1, 1, 1, 3, 3, 1, 3, 1, 1, 2, 1, 1, 1, 1, 1, 2, 3]

```

## 分析数据：使用Matplotlib创建散点图

首先用matplotlib制作原始数据散点图，在控制台输入如下指令：

```python
import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
<matplotlib.collections.PathCollection object at 0x0ED35790>
plt.show()

```

## 准备数据：归一化数值

通过计算距离的方式，我们发现有些差值对计算结果影响非常大，极有可能影响到计算结果

通过式子：newValue = (oldValue - min)/(max-min)

代码如下：

```python
def autoNorm(dataSet):#归一化数值 利用：newValue = (oldValue-min)/(max-min)
    minVals = dataSet.min(0) #不同行比较得出最小数
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet)) #建立与样本相同维度的0矩阵
    m = dataSet.shape[0]  #将样本第一维度数赋值给m
    normDataSet = dataSet -tile(minVals,(m,1))#复制成与输入矩阵同样大小的矩阵
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

```

控制台输入

```python
normMat,ranges,minValues = kNN.autoNorm(datingDataMat)
```

## 测试算法：作为完整程序验证分类器

代码：

```python
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
```

## 约会网站预测函数

```python
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
```

