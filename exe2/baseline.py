from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from operator import methodcaller

from load_data import load_data
from pre_process import pre_process
from sl_rm import *
from configs import *

def cosine(x,y):
	return np.dot(x,y)/np.linalg.norm(x,ord=2)/np.linalg(y,ord=2)

def KNN_recommend(**NN_paras):
	if NN_paras == {}:
		NN_paras = {'n_neighbors':2,'algorithm':'kd_tree'}
	return KNeighborsClassifier(NN_paras)

def KNN_recommend_run(model_name, X_train, X_test, y_train, y_test, paras={}, **NN_paras):
	if NN_paras == {}:
		NN_paras = {'n_neighbors':2,'algorithm':'kd_tree'}
	if paras == {}:
		paras.update(NN_paras)
	result = load_result(model_name, paras)
	if result is None:
		clf = load_model(model_name, paras)
		if clf is None:
			print("can't find clf",model_name)
			clf = KNeighborsClassifier(**NN_paras)
			clf.fit(X_train, y_train)
			save_model(clf, model_name, paras)
		y_pred = clf.predict(X_test)
		sc = clf.score(X_test, y_test)
		# unweighted mean of metrics for labels
		f1_sc = f1_score(y_test, y_pred, average='macro')
		result = {'score': sc, 'f1_score': f1_sc}
		save_result(result, model_name, paras)
		print("{} with {}: score is {}, f1_score is {}".format(
			model_name, paras, sc, f1_sc))
		return clf
	else:
		clf = load_model(model_name, paras)
		sc, f1_sc = result.values()
		print("{} with {}: score is {}, f1_score is {}".format(
			model_name, paras, sc, f1_sc))
		return clf


def KNN_comapre_run(dist_obj, X_train, X_test, y_train, y_test):
	for d in DIST_LIST:
		for n in NEIGHBORS:
			metric = methodcaller(d)(dist_obj)
			metric_params = {}
			if d == "minkowski":
				metric_params.update({'p':3})
			if d == "cosine":
				d = "pyfunc"
				metric_params.update({'func':cosine})
			KNN_recommend_run(COMPARE, X_train, X_test, y_train, y_test, paras={'n':n,'m':d,'p':metric_params}, algorithm='auto', n_neighbors=n, metric=d,metric_params=metric_params)


if __name__ == "__main__":
	X, y = load_data()
	X_train, X_test, y_train, y_test = pre_process(X,y)
	random_M = np.array(load_result("M"))
	if random_M is None:
		random_P = np.random.rand(1,X_train.shape[0])
		random_M = random_P.T*random_P
		save_result("M")
	dist_obj = Distance(X_train,random_M)
	KNN_comapre_run(dist_obj, X_train, X_test, y_train, y_test)
