#!/usr/bin/env python
# -*- coding:UTF-8 -*-
import logRegres
from numpy import *
dataArr, labelMat = logRegres.loadDataSet()
print dataArr
print labelMat

'''梯度上升算法'''
weights1 = logRegres.gradAscent(dataArr, labelMat)
logRegres.plotBestFit(weights1.getA())

'''随机梯度上升算法'''
# print array(dataArr)
weights2 = logRegres.stocGradAscent0(array(dataArr), labelMat)
logRegres.plotBestFit(weights2)

'''改进的随机梯度算法'''
weights3 = logRegres.stocGradAscent1(array(dataArr), labelMat)
logRegres.plotBestFit(weights3)

logRegres.multiTest()