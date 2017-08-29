#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# 使用matplotlib画图

import kNN

# datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*array(datingLabels),15.0*array(datingLabels))
# plt.show()

# normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
# # print normMat
# # print ranges
# # print minVals
# print kNN.datingClassTest()
# print kNN.classifyPerson()

# testVector = kNN.img2vector('testDigits/0_13.txt')
# print testVector[0,0:31]
print kNN.handwritingClassTest()