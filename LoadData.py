# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:18:09 2020

主要工作：
训练集和测试集的划分原则改变了
1.训练集采用数据增强，可手动指定步长，测试集采用无重叠采样的方法
2.训练集 和 测试集 划分的个数可以指定
3.归一化采用 min max归一  逻辑是 训练集产生的min max 作用于 测试集 而不是整个数据集进行归一

存在的问题： 测试集数目太少  还是先数据增强 再划分训练集和测试集试试-------------未完成

准备完成的工作
1. 训练集和测试集 先根据总长度 按 6：4的比例划分
3. 训练集和测试集分别数据增强 ---训练集指定步长  测试集自适应
4. 验证集从测试集中 再划分出来
5. 直接保存 训练集 验证集 和 测试集 的npy文件

@author: zhoubo
"""

import os
from scipy.io  import loadmat
import numpy as np
import random




#

def wgn(x, snr):
    Ps = np.sum(abs(x)**2)/len(x)
    Pn = Ps/(10**((snr/10)))
    noise = np.random.randn(len(x)) * np.sqrt(Pn)
    plt.figure(1)
    plt.plot(noise)
    signal_add_noise = x + noise
    return signal_add_noise




# 写一个自己的dataset类

import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self,filename):
        xy=np.load(filename)
        self.len=xy.shape[0]
        self.x_data=torch.from_numpy(xy[:,:-1]).float().unsqueeze(axis=1)
        self.y_data=torch.from_numpy(xy[:,-1]).int()
    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return self.len

def normalization(data):#归一化函数
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
def data_generator(path,mark):
    """
       读取.mat文件，返回数据的生成器：标签，样本数据。

       :param data_path：.mat文件所在路径
       :param mark：要提取的数据标识"FE" 或 "DE"
       :return generator：返回一个生成器（标签，样本数据）
    """
    # 创建故障和其对应的标签字典
    labels = {"normal":0, "IR007":1, "B007":2, "OR007":3, "IR014":4,
         "B014":5, "OR014":6, "IR021":7, "B021":8, "OR021":9}
    # 列出所有文件
    filenams = os.listdir(path)

    # 逐个对mat文件进行打标签和数据提取
    for single_mat in filenams:
        single_mat_path = os.path.join(path, single_mat)
        # 打标签

        for key, _ in labels.items():
            if key in single_mat:
                label = labels[key]
        # 数据提取
        file = loadmat(single_mat_path)
        for key, _ in file.items():
            if mark in key:
                data = file[key]

        yield label, data

# 数据增强
def  data_augmentation(data_generator,win_len,train_num,test_num,stride,method='0'):
    ''':param  data_generator  数据生成器
       :param  win_len  滑窗的时间长度
       :param

    '''
    #其中训练集数据采用重叠取样  测试集采用无重叠取样
    print('train num:%d\ntest num:%d'%(train_num,test_num))#显示每种故障 训练 测试分别截取多少个
    print('train part overlap rate: %.2f'%((win_len-stride)/win_len))

    train_dataset_list=[]#用来存放每种故障截取出来的数据（数据+标签）
    test_dataset_list=[]

    for label,data in data_generator:

        data=data.reshape(1,-1)#数据展成一行
        length=data.shape[1]# 读取data的长度


        train_part_len=int(length*0.7)#计算用来数据增强的未划分前的训练集长度
        test_part_len=length-train_part_len
        #划分用于切分的训练集 和  测试集的data
        train_part_data=data[:,:train_part_len]
        test_part_data=data[:,train_part_len:]

        max_stride=(train_part_len-win_len)//(train_num-1)

        assert stride < max_stride #保证截取训练集的时候步长要小于所能截取指定训练样本数的最大步长


        #对于训练数据的部分采用 滑动取样
        start_index=0#开始索引为0
        train_data_list=[]#创建一个用来存放切分训练集data后样本的 list容器
        test_data_list=[]
        for i in range(train_num):

            slice_idx =slice(start_index,start_index+win_len)#slice对象  就相当于一个索引
            split_data =train_part_data[:,slice_idx]#切割出来的是split_data （用索引方式获得）
            split_data=split_data.squeeze()#截取的数据降维
            train_data_list.append(split_data)#将切好的data（也就是一个样本）放入容器
            start_index +=stride#下个切割index更新为  经过一个步长后的 位置


        train_samples=np.array(train_data_list)#将截取的列表转换成numpy数组


        # 训练集打标签
        label_cols=np.ones(train_samples.shape[0]).transpose()*label

        #数据和标签组合到一起
        single_train_dataset =np.column_stack((train_samples,label_cols))

        #添加进dataset_list
        train_dataset_list.append(single_train_dataset)

        #开始对训练集部分进行截取
        start_index=0
        for i in range(test_num):
            test_stride =(test_part_len-win_len)//(test_num-1)

            slice_idx =slice(start_index,start_index+win_len)#slice对象  就相当于一个索引
            split_data =test_part_data[:,slice_idx]#切割出来的是split_data （用索引方式获得）
            split_data=split_data.squeeze()#截取的数据降维
            test_data_list.append(split_data)#将切好的data（也就是一个样本）放入容器
            start_index +=test_stride#下个切割index更新为  经过一个步长后的 位置

        test_samples=np.array(test_data_list)
                # 打标签
        label_cols=np.ones(test_samples.shape[0]).transpose()*label

        #数据和标签组合到一起
        single_test_dataset =np.column_stack((test_samples,label_cols))

        #添加进dataset_list
        test_dataset_list.append(single_test_dataset)




    for index,data in enumerate(train_dataset_list,0):
        if index==0:
            train_dataset=data
        else:
            train_dataset=np.vstack((train_dataset,data))

    for index,data in enumerate(test_dataset_list,0):
        if index==0:
            test_dataset=data
        else:
            test_dataset=np.vstack((test_dataset,data))
    # print(train_dataset[0])

    # 对数据集进行归一化
    #归一化应该 将每一类故障 分成训练 测试 部分后 求出 训练的min max 并min max 归一 再把它用到测试上


    if method =='0':#MinMaxScaler归一化
        #对训练集归一化 并计算训练集的 min max值
        X=train_dataset[:,:-1]
        y=train_dataset[:,-1]
        t_min=X.min()
        t_max=X.max()
        X = normalization(X)
        train_dataset=np.column_stack((X,y))#将处理好的X部分与y拼接起来

        #测试集归一化 利用训练集的min max
        X=test_dataset[:,:-1]
        y=test_dataset[:,-1]
        X=(X-t_min)/(t_max-t_min)
        test_dataset=np.column_stack((X,y))

    # print(train_dataset[0])

    #打乱训练集和测试集

    np.random.shuffle(train_dataset)
    np.random.shuffle(test_dataset)

    #保存数据集
    np.save('train_dataset',train_dataset)
    np.save('test_dataset',test_dataset)

    np.save('whole_dataset',np.vstack((train_dataset,test_dataset)))#有可能有用
    # return train_dataset,test_dataset



def get_dataset(path,mark,win_len,train_num,test_num,stride=110,method='0'):
    data_itreror = data_generator(path, mark)
    data_augmentation(data_itreror,win_len,train_num,test_num,stride,method)
    train_db=MyDataset('train_dataset.npy')#实例化自己的dataset
    test_db=MyDataset('test_dataset.npy')#实例化自己的dataset

    return train_db,test_db









if __name__=='__main__':
    # demmo
    np.random.seed(666)
    a,b =get_dataset('./HP0','DE',2048,400,100,125)

    #数据可视化
    import matplotlib.pyplot as plt
    data =a[:5][0].squeeze()
    labels=a[:5][1]
    fig,ax =plt.subplots(nrows=5)

    for i in range(5):
        # ax[i].set_xticks([])#取消x，y刻度
        # ax[i].set_yticks([])

        ax[i].plot(data[i][:2048],label='type:{}'.format(labels[i].item()))
        handle, label = ax[i].get_legend_handles_labels()
        ax[i].legend(handle, label,loc='lower right')
        plt.show()

    plt.tight_layout()




