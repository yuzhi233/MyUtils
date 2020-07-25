# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 10:01:54 2020

@author: zhoubo
"""
from matplotlib import pyplot as plt
import torch
from IPython import display

#------------------------画测试集训练集的loss -epoch 图----------------------------------
def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')
def set_figsize(figsize=(5, 2)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize

#这个可以调用

def draw_loss(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,legend=None, figsize=(3.5, 2.5)):
    plt.figure()
    # plt.tight_layout()
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x_vals, y_vals,color='c',linestyle='-',marker = 'o',markerfacecolor='b',markersize = 3)
    if x2_vals and y2_vals :
        plt.plot(x2_vals, y2_vals,color='m',linestyle='-.',marker = 's',markerfacecolor='r',markersize = 3)
        plt.legend(legend)



def draw_accuracy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,legend=None, figsize=(3.5, 2.5)):
    plt.figure()
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x_vals, y_vals,color='c',linestyle='-',marker = 'o',markerfacecolor='b',markersize = 5)
    if x2_vals and y2_vals :
        plt.plot(x2_vals, y2_vals,color='m',linestyle='-.',marker = 's',markerfacecolor='r',markersize = 5)
        plt.legend(legend)
#------------------------------------------------------------------------------


#----------------------------计算混淆矩阵-----------------------------------------
def confusion_matrix(preds,labels,conf_matrix):#传入的prebs是argmax后的
    preds =preds.int()
    labels =labels.int()
    for p,t in zip(preds,labels):
        conf_matrix[p,t] += 1
    return conf_matrix

if __name__ =='__main__':

    import numpy as np
    import matplotlib.pyplot as plt

    x = np.linspace(-10,10)
    y = np.sin(x)
    plt.plot(x,y,color='m',linestyle='-.',marker = 's',markerfacecolor='r',markersize = 5)
    plt.xlabel('x:---')
    plt.ylabel('y:---')
    plt.title('Titles:---')
    # plt.text(0,0,'Mark')
    plt.grid(True)
    # plt.annotate('focus',xy=(-5,0),xytext=(-2,0.25),arrowprops = dict(facecolor='red',shrink=0.05,headlength= 20,headwidth = 20))








