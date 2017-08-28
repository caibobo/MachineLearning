#!/usr/bin/env python
# -*- coding:UTF-8 -*-

from numpy import *
import operator
from os import listdir

def classify0(inX, dataSet, labels, k):

    #计算距离
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1))-dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}

    #取距离最小的K个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #排序
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = zeros((numberOfLines, 3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return
    fr = open(filename)
    index = 0

    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1

    return returnMat,classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)   #取得每列的最小值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat ,ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :],normMat[numTestVecs:m, :],\
                                     datingLabels[numTestVecs:m], 10)
        print "分类器返回：%d，真实分类是：%d"\
                %(classifierResult,datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "整体错误率为：%f" % (errorCount/float(numTestVecs))

def classifyPerson():
    resultList = ['不喜欢','一般喜欢','很喜欢']
    percentTats = float(raw_input(\
                    "每周玩游戏的时间？"))
    ffMiles = float(raw_input(\
                    "每年飞行的里程？"))
    iceCream = float(raw_input(\
                    "每周冰淇淋消费数？"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    # print inArr
    classifierResult = classify0((inArr-\
                                  minVals)/ranges, normMat, datingLabels, 3)
    # print classifierResult
    print '对这个人的感觉：',\
            resultList[classifierResult-1]

def img2vector(filename):  #图像转换为测试向量
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


def handwritingClassTest(): #手写数字识别
    hwLabels = []
    trainingFileList = listdir('trainingDigits')  #将trainingDigits目录下的文件存储在trainingFileList列表中
    m = len(trainingFileList)               #得到目录下有多少文件，存储为变量m
    trainingMat = zeros((m, 1024))          #创建m行1024列的训练矩阵，每行数据存储一个图像
    for i in range(m):
        '''从文件中解析出分类数字'''
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        '''目录下的文件按照规则命名，比如文件9_45.txt的分类是9，它是数字9的第45个实例'''

        hwLabels.append(classNumStr)  #将类别代码放在hwLables向量中
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr) #载入图像并转化完

    testFileList = listdir('testDigits') #将testDigits目录下的文件存储在testFileList中
    errorCount = 0.0                     #用于计算错误率
    mTest = len(testFileList)            #得到目录中的文件数，存储为mTest
    for i in range(mTest):
        '''从文件中解析出分类数字'''
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        '''作为判断依据'''

        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)  #载入图像并完成转化
        classifierResult = classify0(vectorUnderTest,\
                                     trainingMat,hwLabels, 3)    #使用分类器进行分类
        print "分类器返回：%d,实际类别为：%d"\
                % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\n整体错误为： %d" % errorCount
    print "\n整体错误率为：%f" % (errorCount/float(mTest))

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

