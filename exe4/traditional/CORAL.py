# encoding=utf-8
"""
    Created on 16:31 2018/11/13 
    @author: Jindong Wang
"""
import sys
sys.path.append('..')
import os
import numpy as np
from baseline import SVM_recommend_run, KNN_recommend_run
from load_data import load_data
import scipy.io
import scipy.linalg
import sklearn.metrics
import sklearn.neighbors

class CORAL:
    def __init__(self):
        super(CORAL, self).__init__()

    def fit(self, Xs, Xt):
        '''
        Perform CORAL on the source domain features
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: New source domain features
        '''
        cov_src = np.cov(Xs.T) + np.eye(Xs.shape[1])
        cov_tar = np.cov(Xt.T) + np.eye(Xt.shape[1])
        A_coral = np.dot(scipy.linalg.fractional_matrix_power(cov_src, -0.5),
                         scipy.linalg.fractional_matrix_power(cov_tar, 0.5))
        Xs_new = np.dot(Xs, A_coral)
        return Xs_new

    def fit_predict(self, Xs, Ys, Xt, Yt,**paras):
        '''
        Perform CORAL, then predict using 1NN classifier
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: Accuracy and predicted labels of target domain
        '''
        Xs_new = self.fit(Xs, Xt)
        SVM_recommend_run("CORAL",Xs_new, Xt, Ys, Yt, paras=paras)
        KNN_recommend_run("CORAL",Xs_new, Xt, Ys, Yt, paras=paras)



if __name__ == '__main__':
    os.chdir('..')
    for data in range(3):
        X_src, y_src, X_tgt, y_tgt = load_data(data)
        coral = CORAL()
        coral.fit_predict(X_src, y_src, X_tgt, y_tgt,paras={"data":data})
