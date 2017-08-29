#!/usr/bin/env python
# -*- coding:UTF-8 -*-
#导入开发包
from numpy import *

a = random.rand(4,4)   #产生随机的4*4数组
print a

randMat = mat(a)  #将数组转化为矩阵
b = randMat.I   #求矩阵的逆
print randMat
print b

c = randMat*b  #矩阵与其逆矩阵相乘
print c

d = c-eye(4)  #eye(4)能创建4*4的矩阵
print d
