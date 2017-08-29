#!/usr/bin/env python
# -*- coding:UTF-8 -*-

from numpy import *



'''loadDataSet为便利函数，打开文件testSet.txt并逐行读取
每行前两个值分别是X1和X2，第三个值时数据对应的类别标签
同时为了方便计算，该函数将X0设为1.0'''
def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

'''sigmoid函数'''
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

'''梯度上升算法'''
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)     #将dataMatIn的列表数据转化为矩阵dataMatrix
    # print dataMatrix
    # print mat(classLabels)
    labelMat = mat(classLabels).transpose()   #通过mat（）将classLabels的列表数据转为矩阵，并通过transpose（）进行转置
    # print labelMat
    m, n = shape(dataMatrix)        #m为dataMatrix矩阵的行数，n为dataMatrix矩阵的列数
    # print m
    # print n
    alpha = 0.001           #alpha为步长
    maxCycle = 500          #maxCycle为迭代次数
    weights = ones((n, 1))  #创建n*1的单位矩阵，将回归系数都初始化为1
    # print weights
    for k in range(maxCycle):   #进行迭代，对整个样本集合进行计算
        h = sigmoid(dataMatrix*weights)   #计算sigmoid的输出作为h，输入z为dataMatrix为特征与回归系数相乘
        error = (labelMat - h)            #计算真实类别与预测类别的差值，得到方向
        weights = weights + alpha*dataMatrix.transpose()*error  #以此差值的方向调整回归系数
    return weights

'''绘图'''
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    # weights = wei.getA()
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

'''随机梯度上升算法'''
def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)   #m为矩阵的行数，即样本点个数，n为矩阵的列数，即特征的个数
    alpha = 0.01               #步长
    weights = ones(n)          #初始化回归系数为1,为[1,1,1]
    # print weights
    for i in range(m):         #遍历整个样本点
        h = sigmoid(sum(dataMatrix[i]*weights))  #计算样本点i的sigmoid函数值
        error = classLabels[i] - h               #计算样本点i的真实类别与预测类别的差值
        weights = weights + alpha*error*dataMatrix[i]   #计算样本点i的回归系数
    return weights

'''改进的随机梯度上升算法'''
def stocGradAscent1(dataMatrix, classLabels, numIter=150):  #numIter为迭代次数
    m, n = shape(dataMatrix)    #m为矩阵的行数，即样本点个数，n为矩阵的列数，即特征的个数
    weights = ones(n)           #初始化回归系数为1，[1,1,1]
    for j in range(numIter):    #numIter为迭代次数
        dataIndex = range(m)    #取得样本点的范围
        # print dataIndex
        for i in range(m):      #遍历整个样本点
            '''
            ①步长在每次迭代的时候都会调整，这样会缓解数据波动或者高频波动
            alpha会随着迭代次数不断减小，但永远不会减小到0，因为存在一个常数项
            为了保证在多次迭代之后新数据仍然具有一定的影响
            如果要处理的问题是动态变化的，那么可以适当加大上述常数项，来确保新的值获得更大的回归系数
            在降低alpha的函数中，alpha每次减少1/(j+i)，其中j是迭代次数，i是样本点的下标
            当j<<max(i)时，alpha就不是严格下降的，避免参数的严格下降也常见于模拟退火算法等其他优化算法中
            ②通过选取随机样本来更新回归系数，这种方法将减少周期性的波动
            '''
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(random.uniform(0, len(dataIndex))) #从一个均匀分布[low,high)中随机采样，注意定义域为左闭右开
            h = sigmoid(sum(dataMatrix[randIndex]*weights))    #计算随机样本点randIndex的sigmoid函数值
            error = classLabels[randIndex] - h                 #计算随机样本点randIndex的真实类别与预测类别的差值
            weights = weights + alpha*error*dataMatrix[randIndex]    #计算随机样本点randIndex的回归系数
            del(dataIndex[randIndex])                          #因为randIndex根据dataIndex产生，找到dataIndex[randIndex]删除该值
    return weights

'''用来对样本进行分类，inX为样本特征，weights为回归系数
如果sigmoid函数的值大于0.5，分到1，否则到0'''
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

'''打开训练集和测试集，并对数据进行格式化处理，返回错误率'''
def colicTest():
    '''frTrain打开训练集，frTest打开测试集'''
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():    #遍历训练集
        currLine = line.strip().split('\t') #先对训练集的每行进行格式化处理
        lineArr = []                        #存放样本的特征和类别
        for i in range(21):                 #前20个值为特征
            lineArr.append(float(currLine[i]))   #将特征添加到lineArr列表
        trainingSet.append(lineArr)              #将特征数据添加到训练集列表
        trainingLabels.append(float(currLine[21]))  #将类别添加到训练类别列表
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)     #通过随机梯度算法计算出训练集的每个特征的回归系数
    errorCount = 0; numTestVec = 0.0    #errorCount用于错误计数，numTestVec用于测试集测试样本个数计数
    for line in frTest.readlines():     #遍历测试集
        numTestVec += 1.0               #对测试集中数据的个数计数
        currLine = line.strip().split('\t')     #对测试集的每行进行格式化处理
        lineArr = []                            #存放测试样本的特征和类别
        for i in range(21):                     #遍历所有特征（20个）
            lineArr.append(float(currLine[i]))  #将特征添加到lineArr
        '''如果预测分类与真实类别不一致，错误值加1'''
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)  #计算错误率=错误数/整个测试数
    print "the error rate of this test is : %f" % errorRate
    return errorRate

'''调用colicTest()10次并求结果的平均值'''
def multiTest():
    numTests = 10; errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print "after %d iterations the average error rate is: %f" %(numTests, errorSum/float(numTests))