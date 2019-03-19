from sklearn.feature_selection import (
	VarianceThreshold, SelectKBest, chi2, SelectFromModel)
import numpy as np

from configs import *
from baseline import SVM_recommend

FORWARD_ACCURACY = 0.1

# feature太多，不便于逐个前向/后向筛选，故采用一些筛选的策略


def VT(X, y):
	# 利用方差进行选拔, 这里是backward
	for var in [0.03*i for i in range(1, 10)]:
		selector = VarianceThreshold(threshold=var)
		X = selector.fit_transform(X)
		X_train, X_test, y_train, y_test = pre_process(X, y)
		SVM_recommend(B_VT, X_train, X_test, y_train, y_test)


def UF(X, y):
	# 计算feature间的相关性，进行选择，这里做的是forward
	k_range = [200 for i in range(1, 10)]
	for k in k_range:
		selector = SelectKBest(chi2, k=k)
		X = selector.fit_transform(X)
		X_train, X_test, y_train, y_test = pre_process(X, y)
		SVM_recommend(F_UF, X_train, X_test, y_train, y_test)


def SFM(X, y):
	# 从模型中选择，根据重要性，类似逐个选择，后向选择，逐渐抛弃不重要的
	X_train, X_test, y_train, y_test = pre_process(X, y)
	clf = SVM_recommend(F_UF, X_train, X_test, y_train, y_test)
	m_range = [2000-200*i for i in range(1,10)]
	for m in m_range:
		selector = SelectFromModel(clf,threshold=-np.inf,max_features=m) # 只根据max_features确定选择的数量，不设定threshold
		X = selector.fit_transform(X)
		X_train, X_test, y_train, y_test = pre_process(X, y)
		clf = SVM_recommend(F_UF, X_train, X_test, y_train, y_test)



def reduce(model_name, X, y):
	if model_name == B_VT:
		VT(X, y)
	elif model_name == F_UF:
		UF(X, y)
	elif model_name == B_SFM:
		SFM(X, y)


if __name__ == "__main__":
	X, y = load_data()
	reduce(B_VT, X, y)
	# reduce(F_UF, X, y)
	# reduce(B_SFM, X, y)
