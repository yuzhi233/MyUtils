# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 21:30:37 2020

@author: zhoubo
"""

#coding=utf-8
import matplotlib.pyplot as plt
import numpy as np

confusion = np.array(([51,  0,  0, 0 , 0,  0 , 0,  0 , 0 , 0],
 [ 0, 53 , 0 , 0,  0 , 0,  0 , 0,  0,  0],
 [ 0,  0, 43,  0,  0 , 0,  0 , 0,  0 , 0],
 [ 0,  0,  0, 56,  0 , 0,  0 , 0,  0 , 0],
 [ 0 , 0 , 0,  0, 51 , 0,  0 , 0,  0 , 0],
 [ 0 , 0,  0 , 0 , 0 ,48,  0 , 0,  0 , 0],
 [ 0,  0,  0 , 0 , 0 , 0, 53 , 0,  0 , 0],
 [ 0 , 0 , 0 , 0,  0 , 0,  0 ,47,  0 , 0],
 [ 0 , 0 , 1 , 0 , 0 , 0,  0 , 0 ,54  ,0],
 [ 0 , 0 , 0 , 0 , 0 , 0 , 0  ,0 , 0, 43]))
# 热度图，后面是指定的颜色块，可设置其他的不同颜色
plt.imshow(confusion, cmap=plt.cm.Blues)
# ticks 坐标轴的坐标点
# label 坐标轴标签说明
indices = range(len(confusion))
# 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
#plt.xticks(indices, [0, 1, 2])
#plt.yticks(indices, [0, 1, 2])
plt.xticks(indices, ["NORMAL", "IR007", "B007", "OR007", "IR014","B014", "OR014", "IR021", "B021", "OR021"], rotation=45)
plt.yticks(indices, ["NORMAL", "IR007", "B007", "OR007", "IR014","B014", "OR014", "IR021", "B021", "OR021"])

plt.colorbar()

plt.xlabel('预测值')
plt.ylabel('真实值')
plt.title('混淆矩阵')

# plt.rcParams两行是用于解决标签不能显示汉字的问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 显示数据
for first_index in range(len(confusion)):    #第几行
    for second_index in range(len(confusion[first_index])):    #第几列
        plt.text(first_index, second_index, confusion[first_index][second_index])
# 在matlab里面可以对矩阵直接imagesc(confusion)
# 显示
plt.show()