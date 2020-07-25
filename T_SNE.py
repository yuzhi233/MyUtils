# -*- coding: utf-8 -*-
"""
Created on Sat May  9 09:08:37 2020

@author: zhoubo
"""

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.patheffects as PathEffects

# We import seaborn to make nice plots.
import seaborn as sns
from matplotlib import cm
from sklearn.datasets import load_iris
from mpl_toolkits.mplot3d import Axes3D



def t_sne(a,b):#这个是最基础的不推荐

    X_tsne = TSNE(perplexity=30,n_components=2, learning_rate=120).fit_transform(a)
    X_pca = PCA().fit_transform(a)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=b)
    plt.subplot(122)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=b)





def t_sne2(data,labels):#这个直接显示类别
    X_tsne = TSNE(perplexity=30,n_components=2, learning_rate=120).fit_transform(data)
    plt.cla()
    X,Y=X_tsne[:, 0], X_tsne[:, 1]
    for x, y, s in zip(X, Y, labels):
          c = cm.rainbow(int(255/9 * s)) # 为了使得颜色有区分度，把0-255颜色区间分为9分,然后把标签映射到一个区间
          plt.text(x, y, s, backgroundcolor=c, fontsize=9)


    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max());
    plt.title('Visualize last layer')
    plt.show()
    # plt.savefig("{}.jpg".format(i))

def t_sne2d(features,targets):#10分类 这个是调色盘 推荐这种！！！
    plt.figure()
    embeddings = TSNE(perplexity=30,n_jobs=4,learning_rate=100).fit_transform(features)#n_job是并行处理
    vis_x = embeddings[:, 0]
    vis_y = embeddings[:, 1]
    plt.scatter(vis_x, vis_y, s=50,c=targets, cmap=plt.cm.get_cmap("jet", 10), marker='.')#s是调整点的大小的
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.show()


def t_sne3d(features,targets):
    fig = plt.figure()
    embeddings = TSNE(n_components=3,perplexity=30,learning_rate=150,n_jobs=4).fit_transform(features)#n_job是并行处理
    x_min, x_max = np.min(embeddings, 0), np.max(embeddings, 0)
    embeddings = embeddings / (x_max - x_min)
    vis_x = embeddings[:, 0]
    vis_y = embeddings[:, 1]
    vis_z = embeddings[:, 2]
    ax = Axes3D(fig)
    ax.scatter(vis_x, vis_y, vis_z,s=50,c=targets, cmap=plt.cm.get_cmap("jet", 10), marker='.')#s是调整点的大小的

    # plt.clim(-0.5, 9.5)
    plt.show()


if __name__ =='__main__':

    mydataset =np.load('test_dataset.npy')
    # mydataset =mydataset.astype(np.float32)

    a =mydataset[:,:-1].astype(np.float32)[:100]
    b=mydataset[:,-1].astype(np.int32)[:100]



    # t_sne3(a,b)

    # from mpl_toolkits.mplot3d import Axes3D
    # embedded = TSNE(n_components=3).fit_transform(a)
    # x_min, x_max = np.min(embedded, 0), np.max(embedded, 0)
    # embedded = embedded / (x_max - x_min)
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(embedded[:, 0], embedded[:, 1], embedded[:, 2],s=80,
    #            c=b,cmap=plt.cm.get_cmap("jet", 10), marker='.')
    # plt.clim(-0.5, 9.5)
    # # plt.colorbar(ticks=range(10))
    # plt.show()

    t_sne3d(a,b)


