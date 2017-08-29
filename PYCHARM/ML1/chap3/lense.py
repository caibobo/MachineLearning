#!/usr/bin/env python
# -*- coding:UTF-8 -*-
import trees
import treePlotter

fr = open('lenses.txt')
lense = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lenseTree = trees.creatTree(lense, lensesLabels)

print lenseTree
treePlotter.createPlot(lenseTree)