#!/usr/bin/env python
# -*- coding:UTF-8 -*-
import trees
import treePlotter
myDat, labels = trees.createDataSet()
myTree = treePlotter.retrieveTree(0)
trees.storeTree(myTree, 'classifierStorage.txt')
print trees.grabTree('classifierStorage.txt')
# firstStr = myTree.keys()[0]
# secondDict = myTree[firstStr]
# print secondDict.keys()
# for key in secondDict.keys():
#     print key
# print secondDict
# print s2
# featIndex = labels.index(firstStr)
# print featIndex
# featIndex2=labels.index(s2)
# print featIndex2
# # list1= [1,1,2,3]
# # print list1[:0]
# print myDat
# print trees.chooseBestFeatureTopSplit(myDat)
# print myTree
# # print trees.calcShannonEnt(myDat)
# # print trees.splitDataSet(myDat, 0, 1)
# # print trees.splitDataSet(myDat, 0, 0)