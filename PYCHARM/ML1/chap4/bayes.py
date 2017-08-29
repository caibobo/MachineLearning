#!/usr/bin/env python
# -*- coding:UTF-8 -*-
from numpy import *

def loadDataSet():   #载入词数据
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def createVocabList(dataSet):   #创建一个包含在所有文档中出现的不重复词的列表
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):  #词集模型，将词转化为向量，生成每个词是否出现的列表
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word : %s is not in my Vocabulary!" % word
    return returnVec

def trainNB0(trainMatrix, trainCategory):   #朴素贝叶斯分类器训练函数
    numTrainDocs = len(trainMatrix)         #表示训练矩阵的长度即有多少组
    # print numTrainDocs
    numWords = len(trainMatrix[0])          #因为训练矩阵表示每个词是否出现,为[[],[],....,[]]形式，其中每个[]里的长度即为整个单词数
    # print trainMatrix[0]
    # print numWords
    pAbusive = sum(trainCategory)/float(numTrainDocs)  #属于侮辱性文档的概率，trainCategory中的1表示为侮辱性文档，numTrainDocs表示整个文档数目
    # print trainCategory
    # print sum(trainCategory)
    # print float(numTrainDocs)
    # print pAbusive
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    # print p0Num,p1Num
    p0Denom = 2.0; p1Denom = 2.0
    for i in range(numTrainDocs):            #遍历所有文档
        if trainCategory[i] == 1:            #如果文档为侮辱性文档
            p1Num += trainMatrix[i]          #计算各个词汇的出现次数矩阵
            p1Denom += sum(trainMatrix[i])   #计算整个文档中词的次数
        else:                                #如果文档为非侮辱性文档
            p0Num += trainMatrix[i]          #计算各个词汇的出现次数矩阵
            p0Denom += sum(trainMatrix[i])   #计算整个文档中词的次数


    # print p0Denom
    # print p1Num
    p1Vect = log(p1Num/p1Denom)     #change to log()避免下溢出
    p0Vect = log(p0Num/p0Denom)     #change to log()
    # print p0Vect
    # print p1Vect
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    '''p1为贝叶斯公式计算出的在有这些词的情况下文档属于侮辱性文档的概率'''
    p1 = sum(vec2Classify*p1Vec) + log(pClass1)

    # print vec2Classify
    #
    # print vec2Classify*p1Vec
    # print sum(vec2Classify*p1Vec)
    # print p1
    '''p0为贝叶斯公式计算出的在有这些词的情况下
        文档属于侮辱性文档的概率'''
    p0 = sum(vec2Classify*p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():            #便利函数，封装了所有操作
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList()
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)

def bagOfWords2VecMN(vocabList, inputSet):   #词袋模型，计算词在文档中出现的次数
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

'''
垃圾邮件过滤的相关函数
'''
def textParse(bigString):                #进行切分文本，对文本解析
    import re
    listOfTokens = re.split(r'\W*', bigString)  #利用正则表达式切分文本，其中\W表示匹配非字母数字以及下划线（即不匹配字母数字以及下划线）
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]  #将字符串长度少于2的字符去除，返回字符串列表

def spamTest():                 #对贝叶斯垃圾邮件分类器自动化处理
    '''docList是email/spam下的1-26个txt文件中每个文件的词，每个文件中的词集构成此列表中的每项元素
    classList是邮件分类列表，1表示垃圾邮件，0表示正常邮件
    fullText是email/spam下的1-26个txt文件中总共出现的词'''
    docList = []; classList = []; fullText = []
    for i in range(1,26):
        # 将email/spam下的1-26个txt文件每个文件的内容文本解析，返回为wordList，其中包含i.txt中的词
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        # print wordList
        docList.append(wordList)  #将i.txt中的词列表以append方式添加到docList中
        # print docList
        fullText.extend(wordList) #将i.txt中的词列表以extend方式添加到fullText中
        # print fullText
        classList.append(1)       #因为是垃圾邮件，所以将类列表中添加1，表示为垃圾文档
        # print classList
        # 将email/ham下的1-26个txt文件每个文件的内容文本解析，返回为wordList，其中包含i.txt中的词
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)        #将i.txt中的词列表以append方式添加到docList中
        fullText.extend(wordList)       #将i.txt中的词列表以extend方式添加到fullText中
        classList.append(0)             #因为是正常邮件，所以将类列表中添加0，表示为正常文档
    # print docList
    # print classList
    vocabList = createVocabList(docList)    #将docList中的词生成无重复的词列表
    trainingSet = range(50); testSet = []   #trainingSet是训练集[0,1,2,...,49]，testSet为测试集
    # print trainingSet
    for i in range(10):                      #随机选10封邮件作为测试集
        '''产生选中邮件的索引
        random.uniform()表示从一个均匀分布[low,high)中随机采样，注意定义域为左闭右开'''
        randIndex = int(random.uniform(0,len(trainingSet)))

        '''将索引为randIndex的trainingSet中文件添加到testSet'''
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])      #从训练集中删除选到测试集的文件
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:         #遍历整个训练集
        '''将docList[docIndex]中的词转换为向量，
            即将docList中在训练集中的项中的词转换为向量，
            然后添加到trainMat，同时将其类别添加到trainClasses中'''
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    '''p0V表示为正常邮件的条件下每个词出现的概率
        p1V表示为垃圾邮件的条件下每个词出现的概率
        pSpam表示垃圾邮件的概率'''
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:    #遍历测试集
        '''将docList[docIndex]中的词转换为向量，
            即将docList中在测试集中的项中的词转换为向量'''
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])

        '''如果朴素贝叶斯分类器计算出来的类别和该邮件真正所属类别不一致，
            则错误数加1，并且输出错误的文档的词列表'''
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print "classification error", docList[docIndex]
    print 'the error rate is: ', float(errorCount)/len(testSet)

'''使用朴素贝叶斯分类器从个人广告中获取区域倾向
    calcMostFreq计算高频词，返回前30个高频词'''
def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}           #创建高频词字典
    for token in vocabList: #遍历整个vocabList中的词
        freqDict[token] = fullText.count(token)   #在所给文本中将对应词的次数进行统计，形成freqDict字典的键值对（即词和词的次数）
    '''排序，sorted函数的原型为
    sorted(iterable[,cmp[,key[,reverse]]])
    参数解释
    iterable指定要排序的list或者iterable
    cmp为函数，指定排序时进行比较的函数，可以指定一个函数或者lambda函数
    key为函数，指定取待排序元素的哪一项进行排序
    reverse是一个bool变量，表示升序排列(false)还是降序排列(True)
    freqDict.iteritems()将字典中的所有项，以迭代器的方式返回
    operator模块提供的itemgetter函数用于获取对象的哪些维的数据，参数为一些序号（即需要获取的数据在对象中的序号）
    比如下面就是获取对象也就是字典中的键值对的值
    reverse表示逆序排列，即从高到低，所以取前30个高频词'''
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1),\
                        reverse=True)
    return sortedFreq[:30]

'''localWords函数将两个RSS源作为参数，RSS源要在函数外导入，
    这样做是因为RSS会随时间而改变（如果想通过改变代码来比较程序的差异，就应该使用相同的输入）'''
def localWords(feed1, feed0):
    import feedparser      #导入RSS程序库
    '''
    需要修改
    docList是email/spam下的1-26个txt文件中每个文件的词，每个文件中的词集构成此列表中的每项元素
    classList是RSS分类列表，1表示垃圾邮件，0表示正常邮件
    fullText是email/spam下的1-26个txt文件中总共出现的词'''
    docList = []; classList = []; fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))

    for i in range(minLen):
        '''textParse对feed1和feed0中的进行切分文本
        然后分别将feed1和feed0['entries'][i]['summary']中的词列表以append方式添加到docList中
        将feed1和feed0['entries'][i]['summary']中的词列表以extend方式添加到fullText中
        同时vocabList创建无重复的整个词列表
        top30Words计算出30个高频词并返回'''
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)

    '''遍历整个高频词表，将其从整个单词列表中去除
        traininSet为训练集，testSet为测试集'''
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = range(2*minLen); testSet = []

    '''留存交叉验证
       从数据中选取20个进入测试集，并将其从训练集中删除'''
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet))) #从一个均匀分布[low,high)中随机采样，注意定义域为左闭右开
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []; trainClasses = []

    '''遍历整个训练集，通过词袋模型将词转为向量，将其加入到trainMat，并且将对应的类别加入到trainClasses'''
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    '''p0V为feed0中每个词出现的概率
        p1V为feed1中每个词出现的概率
        pSpam为？概率
        errorCount为错误数'''
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0

    '''遍历测试集，通过词袋模型将词转换为向量，通过classifyNB函数预测分类，如果和实际所属类别不同
        则增加错误数并且输出预测错误的文档以及错误率，并返回整个vocabList, p0V和p1V'''
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList , docList[docIndex])
        if classifyNB(wordVector, p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print "classification error", docList[docIndex]
    print 'the error rate is: ', float(errorCount)/len(testSet)
    return vocabList, p0V, p1V

'''从两个RSS源中取出出现概率大于某个阈值的词'''
def getTopWords(ny, sf):
    import operator
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []; topSF = []
    '''遍历整个词表，将出现概率值大于-6.0的以（词，其概率值）的形式分别添加到topSF和topNY中'''
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i], p1V[i]))
    '''对topSF按照词出现概率的值的大小按从大到小的规则进行排序'''
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF"
    '''遍历已经排序的sortedSF，符合条件的高频词都被输出'''
    for item in sortedSF:
        print item[0]
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY"
    for item in sortedNY:
        print item[0]