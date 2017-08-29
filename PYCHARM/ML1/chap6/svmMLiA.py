#!/usr/bin/env python
# -*- coding:UTF-8 -*-
from numpy import *


'''辅助函数loadDataSet()/selectJrand()/clipAlpha()'''
def loadDataSet(fileName):      #从文件载入数据集
    dataMat = []; labelMat = []         #dataMat为数据矩阵，labelMat为类别矩阵
    fr = open(fileName)                 #打开文件
    for line in fr.readlines():         #遍历文件的每行
        lineArr = line.strip().split('\t')          #对文件进行拆分处理，格式化处理，按照制表符，切割字符串
        dataMat.append([float(lineArr[0]), float(lineArr[1])])  #将数据的前两列加入数据矩阵
        labelMat.append(float(lineArr[2]))              #将数据的最后一列加入类别矩阵
    return dataMat, labelMat                #返回数据矩阵和类别矩阵

def selectJrand(i, m):  #i为alpha下标，m为所有alpha的个数
    j = i
    print i
    print j
    while(j==i):            #当
        j = int(random.uniform(0, m))       #只要函数不等于输入值i,函数进行随机选择
    return j                           #返回另一个下标j

def clipAlpha(aj,H,L):          #用于调整大于H或者小于L的alpha值
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj



'''简化版SMO算法smoSimple
共有五个参数
数据集dataMatIn
类别标签classLabels
常数C
容错率toler
退出前最大的循环次数maxIter
只有在所有数据集上遍历maxIter次，且不再发生任何alpha修改之后，程序才会停止并退出while循环'''
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    '''将dataMatIn转换成Numpy矩阵dataMatrix，将classLabels转换为Numpy矩阵并转置为labelMat'''
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()

    '''b为分隔超平面公式中的参数
    m为dataMatrix的行数
    n为dataMatrix的列数
    alphas为列矩阵，元素都被初始化为0
    iter存储的是在没有任何alpha改变的情况下遍历数据集的次数
    当该变量达到输入值matIter时，函数结束运行'''
    b = 0; m, n = shape(dataMatrix)
    alphas = mat(zeros((m, 1)))
    iter = 0
    while(iter < maxIter):   #未到达退出前最大的循环次数maxIter的条件时
        alphaPairsChanged = 0       #alphaPairsChanged记录alpha是否已经优化，即一对alpha对改变的次数
        for i in range(m):          #遍历整个数据集

            '''fXi是预测i的类别
            Ei为误差'''
            fXi = float(multiply(alphas, labelMat).T*\
                        (dataMatrix*dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMat[i])           #预测类别与真实类别的差值

            '''如果误差很大就对对应的alpha的值进行优化
            在if语句中正间隔还是负间隔都会被测试
            并且在该if语句中，同时要检查alpha的值，以保证其不能等于0或C
            由于后面alpha小于0或大于C时将被调整为0或C，所以一旦if中alpha已经等于0或C
            意味着它们已经在边界上，不再能够减小或增大，也就不值得再对其优化'''
            if ((labelMat[i]*Ei < - toler) and (alphas[i] < C)) or \
                    ((labelMat[i]*Ei > toler) and \
                             (alphas[i] > 0)):
                j = selectJrand(i, m)           #通过辅助函数selectJrand进行另一个alpha值的计算

                '''fXj是预测j的类别
                Ej是误差'''
                fXj = float(multiply(alphas, labelMat).T*\
                            (dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])

                '''将之前的值分别保存在alphaIold、alphaJold'''
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                '''计算L和H，用于将alpha[j]调整到0到C之间，如果L与H相等就不做任何改变
                直接执行continue，也就是直接运行下一次for的循环'''
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C +alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, C + alphas[j] - alphas[i])
                if L == H:
                    print "L == H";continue

                '''eta是alpha[j]的最优修改量，如果为0，实际需要计算新的alpha[j]的值
                但是此处为了方便直接退出当前循环，eta不为0的话，
                改变alpha[j]，使之减小'''
                eta = 2.0*dataMatrix[i, :]*dataMatrix[j, :].T - \
                    dataMatrix[i, :]*dataMatrix[i, :].T - \
                    dataMatrix[j, :]*dataMatrix[j, :].T
                if eta >= 0: print "eta >= 0"; continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[i] = clipAlpha(alphas[j], H, L)

                '''对调整后的alpha进行计算，观察其变化状况的程度，
                如果变化很轻微，退出当前循环
                同样改变alpha[i]，但是与alpha[j]的方向相反，即使之增加'''
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print "j not moving enough"; continue
                alphas[i] += labelMat[j]*labelMat[i]*\
                             (alphaJold - alphas[j])
                '''给alpha[i]与alpha[j]设置好常数项'''
                b1 = b - Ei - labelMat[i]*(alphas[i] - alphaIold)* \
                    dataMatrix[i, :]*dataMatrix[i, :].T - \
                    labelMat[j]*(alphas[j] - alphaJold)* \
                    dataMatrix[i, :]*dataMatrix[j, :].T
                b2 = b - Ei - labelMat[i]*(alphas[i] - alphaIold)* \
                    dataMatrix[i, :]*dataMatrix[j, :].T - \
                    labelMat[j]*(alphas[j] - alphaJold)* \
                    dataMatrix[j, :]*dataMatrix[j, :].T
                '''当alpha[i]或alpha[j]处于0到C之间设置常数项'''
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2.0
                '''执行完for循环的最后一行都不执行continue语句，
                即已经成功改变了一对alpha，增加alphaPairsChanged'''
                alphaPairsChanged += 1
                print "iter: %d i: %d, pairs changed %d" % \
                      (iter, i, alphaPairsChanged)
            '''在for循环之外检查alpha值是否做了更新，
            如果更新则将iter设置为0后继续运行程序'''
            if (alphaPairsChanged == 0):
                iter += 1
            else:
                iter = 0
            print "iteration number: %d" % iter
        return b, alphas

'''核转换函数
三个输入参数，两个数值型变量和一个元组kTup
元组kTup给出的是核函数的信息
元组的第一个参数是描述所用核函数类型的一个字符串，其他2个参数是核函数可能需要的可选参数'''
def kernelTrans(X, A, kTup):
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if kTup[0] == 'lin': K = X*A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2))
    else: raise NameError('Houston We have a Problem -- That Kernel is not recognized')  #遇到一个无法识别的元组，程序抛出异常
    return K

'''完整的Platt SMO算法的支持函数'''
'''建立一个数据结构来保存所有的重要值
构建仅包含init方法的optStruct类，该方法可以实现其成员变量的填充
增加了一个m*2的矩阵eCache，eCache的第一列给出的是eCache是否有效的标志位
第二列是实际的E值'''
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):  #kTup为包含核函数信息的元组
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))
        self.K = mat(zeros((self.m, self.m)))           #先创建矩阵K
        '''利用核转换函数kernelTrans()对矩阵K进行填充'''
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)

'''calcEkK计算k的预测类别与真实类别的误差然后返回该误差'''
def calcEkK(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T* \
                (oS.X*oS.X[k, :].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

'''用于选择第二个alpha的值
这里采用最大步长化的方法，选择步长最大即Ei-Ej最大的alpha值
'''
def selectJ(i, oS, Ei):
    maxK = -1; maxDeltaE = 0; Ej = 0    #
    oS.eCache[i] = [1, Ei]              #将输入值Ei在缓存中设置成有效的，即Ei已经计算好
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]     #构建非零表
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i: continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            '''选择步长最大的j'''
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

'''计算误差值并存入缓存中，再对alpha值优化后会用到这个值'''
def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

'''
完整SMO算法的优化例程
与smoSimple基本相同
但是该代码用了自己的数据结构，该结构参数oS中传递
第二个重要的修改就是使用了selectJ()而不是selectJrand()来选择第二个alpha值
在alpha值改变时更新Ecache'''
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < - oS.tol) and (oS.alphas[i] < oS.C)) or \
            ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy();
        alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H: print "L==H"; return 0
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
        if eta >= 0: print "eta>=0"; return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])  # update i by the same amount as j
        updateEk(oS, i)  # added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T - oS.labelMat[j] * (
        oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - oS.labelMat[j] * (
        oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

'''完整的SMO算法
oS利用optStruct构建了一个数据结构来存放所有数据，然后对需要控制函数退出的一些变量进行初始化'''
def smoPK(dataMatIn, classLabels, C, toler, maxIter):    #full Platt SMO
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler)
    iter = 0
    entireSet = True; alphaPairsChanged = 0

    '''当迭代次数超过指定的最大值或者遍历整个集合都未对任意的alpha进行修改时，就退出循环
    这里的maxIter的作用与之前的smoSimple中有所不同，一开始的for循环在数据集上遍历任意可能的alpha
    通过调用innerL（）来选择第二个alpha，并在可能时对其进行优化处理，
    如果有任意一对alpha值发生了改变，那么就会返回1.
    第二个for循环遍历所有的非边界alpha值也就是不在边界0或C上的值
    接下来我们对for循环在非边界循环和完整遍历之间进行切换，并打印出迭代次数'''
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)
                print "fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print "non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True
        print "iteration number: %d" % iter
    return oS.b,oS.alphas

'''计算w'''
def calcWs(alphas,dataArr,classLabels):
    X = mat(dataArr); labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w


'''核函数版本'''
def innerLK(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print "L==H"; return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #changed for kernel
        if eta >= 0: print "eta>=0"; return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

'''calcEk计算k的预测类别与真实类别的误差然后返回该误差'''
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    #full Platt SMO
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)
                print "fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print "non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True
        print "iteration number: %d" % iter
    return oS.b,oS.alphas

'''径向基测试函数testRbf()
有一个用户定义的变量
整个代码主要由以前定义的函数集合构成
首先从文件读取数据集然后运行SMO算法
其中核函数的类型为rbf
优化过程结束后，在后面的矩阵数学运算中建立了数据的矩阵副本，
并且找出那些非零的alpha值，从而得到所需的支持向量
即得到了这些向量和alpha的类别标签值'''
def testRbf(k1=1.3):
    dataArr, labelArr = loadDataSet('testSetRBF.txt')
    b, alphas =smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    datMat = mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A>0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd];
    print "there are %d Support Vectors" % shape(sVs)[0]
    m, n = shape(datMat)
    errorCount = 0
    '''如何利用核函数分类，首先使用kernelTrans函数得到转换后的数据
    再用其与前面的alpha以及类别标签值求积'''
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :],('rbf', k1))
        predict = kernelEval.T*multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print "the training error rate is: %f" % (float(errorCount)/m)
    dataArr, labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat = mat(dataArr); labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T*multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print "the test error rate is: %f" % (float(errorCount)/m)


'''基于SVM的手写识别问题'''

def img2vector(filename):  #图像转换为测试向量
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    from os import listdir      #listdir用于获得指定目录中的内容
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))   #构建m*1024的训练矩阵
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)
        else: hwLabels.append(1)
        trainingMat[i, :] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels

def testDigits(kTup=('rbf', 10)):
    dataArr, labelArr = loadImages('trainingDigits')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat = mat(dataArr); labelMat = mat(labelArr).transpose()
    '''返回数组a中值不为零的元素的下标，
    它的返回值是一个长度为a.ndim（数组a的轴数）的元组，
    元组的每个元素都是一个整数数组，其值为非零元素的下标在对应轴上的值'''
    svInd = nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print "there are %d Support Vectors" % shape(sVs)[0]
    m, n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :],kTup)
        predict = kernelEval.T*multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print "the training error rate is : %f" % (float(errorCount)/m)
    dataArr, labelArr = loadImages('testDigits')
    errorCount = 0
    datMat = mat(dataArr); labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :],kTup)
        predict = kernelEval.T*multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print "the test error rate is : %f" % (float(errorCount)/m)




