from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from load_data import load_data
from pre_process import pre_process
from sl_rm import *
from configs import *

# 注意weights使用distance时，会取距离倒数为权重，此时使用更多的邻居数目，也可能不会对分类造成影响，因为最近的那个可能永远拥有最大的影响力
# 如果设置uniform是只看数量进行估计
def cosine(x,y):
	s = np.linalg.norm(x,ord=2)*np.linalg.norm(y,ord=2)
	if s==0:
		return 0
	return np.dot(x,y)/s

def KNN_recommend(**NN_paras):
	if NN_paras == {}:
		NN_paras = {'n_neighbors':BSET_N, 'metric':'euclidean', 'algorithm':'auto', 'weights':'uniform'}
	return KNeighborsClassifier(NN_paras)

def KNN_recommend_run(model_name, X_train, X_test, y_train, y_test, bStore=False, paras={}, **NN_paras):
	if NN_paras == {}:
		NN_paras = {'n_neighbors':BSET_N, 'metric':'euclidean', 'algorithm':'auto', 'weights':'uniform'}
	if paras == {}:
		paras.update(NN_paras)
	result = load_result(model_name, paras)
	if result is None:
		clf = load_model(model_name, paras)
		if clf is None:
			print("can't find clf",model_name)
			clf = KNeighborsClassifier(**NN_paras)
			print(clf)
			clf.fit(X_train, y_train)
			if bStore:
				save_model(clf, model_name, paras)
		sc = clf.score(X_test, y_test)
		# unweighted mean of metrics for labels
		result = {'score': sc}
		save_result(result, model_name, paras)
		print("{} with {}: score is {}".format(
			model_name, paras, sc))
		return clf
	else:
		clf = load_model(model_name, paras)
		sc = result.values()
		print("{} with {}: score is {}".format(
			model_name, paras, sc))
		return clf


def KNN_comapre_run(X_train, X_test, y_train, y_test,model_name=COMPARE):
	p = -1
	for d in DIST_LIST:
		for n in NEIGHBORS:
			metric_params = {}
			if d == "minkowski":
				p = 3
				metric_params.update({'m':d,'p':3})
			if d == "cosine":
				d = cosine
				metric_params.update({'m':"cosine"})
			else:
				metric_params.update({'m':d})
			KNN_recommend_run(model_name, X_train, X_test, y_train, y_test, paras={'n':n,**metric_params}, algorithm='auto', n_neighbors=n, metric=d, p=p, weights='uniform')


if __name__ == "__main__":
	X, y = load_data()
	X_train, X_test, y_train, y_test = pre_process(X,y)
	KNN_comapre_run(X_train, X_test, y_train, y_test)
