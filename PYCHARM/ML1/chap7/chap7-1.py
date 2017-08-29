#!/usr/bin/env python
# -*- coding:UTF-8 -*-
import adaboost
from numpy import *

datArr, labelArr = adaboost.loadSimpData()
classifierArr = adaboost.adaBoostTrainDS(datArr, labelArr, 30)
print type(classifierArr)
adaboost.adaClassify([[5, 5], [0, 0]], classifierArr)