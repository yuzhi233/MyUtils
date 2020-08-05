# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 21:54:06 2020
计算时域指标
@author: zhoubo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def get_time_features(data):
    '''输入数据为numpy数组 行为样本数量'''
    xp=data.max(axis=1)# 峰值
    mean=data.mean(axis=1)#均值
    var=data.var(axis=1)#方差
    std=data.std(axis=1)#标准差
    rms=np.sqrt(mean**2+var)#均方值=sqrt(e(x^2))=sqrt(e(x)^2+d(x))
    avg =  np.mean(np.abs(data), axis=1)#整流平均值
    #无量纲值
    skew=pd.DataFrame(data).skew(axis=1).values#偏度 numpy没有计算偏度的利用panda库计算
    kurt=pd.DataFrame(data).kurt(axis=1).values#峭度 numpy没有计算偏度的利用panda库计算
    lp =xp/rms#峰值因子
    cf=xp/avg#脉冲因子
    sf=rms/avg# 波形因子
    ce =xp/(np.mean(np.sqrt(np.abs(data)),axis=1))**2# 裕度因子
    k3=np.mean((data-mean.reshape(-1,1))**3,axis=1)/std**3 #偏度因子
    k4=np.mean((data-mean.reshape(-1,1))**4,axis=1)/std**4#峭度因子


    time_features=np.vstack((xp,mean,var,std,rms,avg,skew,kurt,lp,cf,sf,ce,k3,k4)).transpose(1,0)

    return time_features