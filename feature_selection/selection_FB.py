import sys
sys.path.append('..')

from sklearn.feature_selection import (
	VarianceThreshold, SelectKBest, chi2, SelectFromModel)
import numpy as np
import os

from configs import *
from baseline import SVM_recommend_run, SVM_recommend
from load_data import load_data
from pre_process import pre_process

# feature太多，不便于逐个前向/后向筛选，故采用一些筛选的策略
def VT(X, y):
	# 利用方差进行选拔, 这里是backward
	for var in [0.03*i for i in range(1, 50)]:
		selector = VarianceThreshold(threshold=var)
		X_ = selector.fit_transform(X)
		# X.shape[1] # selected feature amount
		X_train, X_test, y_train, y_test = pre_process(X_, y, bReset=True)
		# 有问题
		SVM_recommend_run(B_VT, X_train, X_test, y_train, y_test, paras={'variance':var,'feature-num':X_.shape[1]})


def UF(X, y):
	# 计算feature间的相关性，进行选择，这里做的是forward
	k_range = [50*i for i in range(1, 4)]
	for k in k_range:
		selector = SelectKBest(chi2, k=k)
		X_ = selector.fit_transform(X,y)
		X_train, X_test, y_train, y_test = pre_process(X_, y, bReset=True)
		SVM_recommend_run(F_UF, X_train, X_test, y_train, y_test, paras={'k-best':k})


def SFM(X, y):
	# 从模型中选择，根据重要性，类似逐个选择，后向选择，逐渐抛弃不重要的
	X_train, X_test, y_train, y_test = pre_process(X, y)
	clf = SVM_recommend()
	m_range = [2000-50*i for i in range(36,40)]
	for m in m_range:
		selector = SelectFromModel(clf,threshold=-np.inf,max_features=m) # 只根据max_features确定选择的数量，不设定threshold
		X_ = selector.fit_transform(np.asarray(X),np.asarray(y))
		X_train, X_test, y_train, y_test = pre_process(X_, y, bReset=True)
		clf = SVM_recommend_run(F_UF, X_train, X_test, y_train, y_test, paras={'max-features':m})



def reduce(model_name, X, y):
	if model_name == B_VT:
		VT(X, y)
	elif model_name == F_UF:
		UF(X, y)
	elif model_name == B_SFM:
		SFM(X, y)


if __name__ == "__main__":
	os.chdir('..')
	X, y = load_data()
	# reduce(B_VT, X, y)
	# reduce(F_UF, X, y)
	reduce(B_SFM, X, y)
