#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# a1 = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]
# a2 = [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
# a3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1]
# print a1+a2+a3
# from numpy import *
# b = log(0.07692308)
# print b
import re
mySent = 'This book is the best book on Python or M.L. I have never laid eyes upon. '
regEx = re.compile('\\W*')
# listOfTokens = regEx.split(mySent)
# print [tok.lower() for tok in listOfTokens if len(tok) > 0 ]

emailText = open('email/ham/6.txt').read()
listOfTokens = regEx.split(emailText)
print listOfTokens