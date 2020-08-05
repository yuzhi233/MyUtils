# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:40:31 2020

@author: zhoubo
"""

import itertools
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    # plt.savefig('confusion_matrix',dpi=200)

cnf_matrix =np.array(([50, 0,  0, 0 , 0,  0 , 0,  0 , 0 , 0],
                     [ 0, 51 , 0 , 0,  0 , 0,  0 , 0,  0, 0],
                     [ 0, 0, 47,  0,  0 , 0,  0 , 0,  0 , 0],
                     [ 0, 0,  0, 45,  0 , 0,  0 , 0,  0 , 0],
                     [ 0, 0 , 0,  0, 51 , 0,  0 , 0,  0 , 0],
                     [ 0, 0,  0 , 0 , 0 ,46,  0 , 0,  0 , 0],
                     [ 0, 0,  0 , 0 , 0 , 0, 55 , 0,  1 , 0],
                     [ 0, 0 , 0 , 0,  0 , 0,  0 ,49,  0 , 0],
                     [ 0, 0 , 0 , 0 , 0 , 0,  0 , 0 ,55  ,0],
                     [ 0, 0 , 0 , 0 , 0 , 0 , 0  ,0 , 0, 50]))

class_names = ["NORMAL", "IR007", "B007", "OR007", "IR014","B014", "OR014", "IR021", "B021", "OR021"]

# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names,
#                       title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')