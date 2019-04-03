
#import sys
#sys.path.append('..')
from time import time
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
import numpy as np
from sklearn import manifold
import os
import matplotlib.pyplot as plt
from baseline import SVM_recommend_run
from pre_process import pre_process
from configs import *
import pylab

def load_label():
    file_name = "./data/AwA2-labels.txt"
    with open(file_name) as f:
        lines = f.readlines()
    y = [int(line.strip('\n')) for line in lines]
    return y


def load_feature():
    file_name = "./data/AwA2-features.txt"
    with open(file_name) as f:
        lines = f.readlines()
    lines = [line.strip('\n') for line in lines]
    X = [list(map(float, line.split(' '))) for line in lines]
    return X


def load_data():
    X = load_feature()
    y = load_label()
    return [X, y]



n_components = 3
perplexity = 50
NUM_COLORS = 60

cm = pylab.get_cmap('gist_rainbow')

if __name__ == "__main__":
    X, y = load_data()
    #print(X.shape)
    print(len(X))
    for n_components in range(2,4):
        for perplexity in range(20,50,5):
            for random_state in range(0,2):
                t0 = time()
                tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=random_state, perplexity=perplexity)
                X_ = tsne.fit_transform(X)
                print(len(X_))
                print(len(X_[0]))
                np.save("data/tsne",X_)
                t1 = time()  # 修改
                print("t-SNE: %.2g sec" % (t1 - t0))
                x_min, x_max = X_.min(0), X_.max(0)
                X_norm = (X_ - x_min) / (x_max - x_min)  # 归一化
                plt.figure(figsize=(8, 8))
                for i in range(X_norm.shape[0]):
                    #plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
                           # fontdict={'weight': 'bold', 'size': 9})
                    plt.scatter(X_norm[i, 0], X_norm[i, 1],color=cm(1. * y[i] / NUM_COLORS))
                plt.xticks([])
                plt.yticks([])
                name = str(n_components) + "  " + str(perplexity) + " " + str(random_state)+".png"
                plt.savefig(name)
                plt.show()
                #X_ = transform(X, y)
                X_train, X_test, y_train, y_test = pre_process(X_, y, bReset=True)
                SVM_recommend_run(tSNE, X_train, X_test, y_train, y_test, paras={'n_cp':n_components,'ppl':perplexity,'rd':random_state})
