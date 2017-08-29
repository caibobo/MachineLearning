#!/usr/bin/env python
# -*- coding:UTF-8 -*-
import kNN
group,labels = kNN.createDataSet()

datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
print group
print labels
print kNN.classify0([0, 0], group, labels, 3)
print datingDataMat
print datingLabels[0:20]