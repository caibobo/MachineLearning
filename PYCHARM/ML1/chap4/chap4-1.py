#!/usr/bin/env python
# -*- coding:UTF-8 -*-
import bayes
import feedparser
# listOPosts, listClasses = bayes.loadDataSet()
# # print listOPosts
# # print listClasses
# #
# myVocabList = bayes.createVocabList(listOPosts)
# #
# print myVocabList
# print bayes.setOfWords2Vec(myVocabList, listOPosts[0])
#
# trainMat = []
# for postinDoc in listOPosts:
#     trainMat.append(bayes.setOfWords2Vec(myVocabList, postinDoc))
# print trainMat
#
# p0V, p1V, pAb = bayes.trainNB0(trainMat, listClasses)
# print p0V
# print p1V

# print bayes.testingNB()
print bayes.spamTest()

ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')