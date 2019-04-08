import sys
#sys.path.append('..')

from numpy import *
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from configs import *
from baseline import SVM_recommend_run
from sklearn.decomposition import PCA
from load_data import *
from pre_process import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC
# feature太多，不便于逐个前向/后向筛选，故采用一些筛选的策略
def apply_lda(a,X_train, X_test, y_train, y_test):
	print("start lda")
	print(a)
	#X_train, X_test, y_train, y_test = pre_process(X, y)
	# a,b,c=X_train.shape
	# print(a)
	# print(b)
	# print(c)
	# X_train=reshape((a,b*c))
	# a,b,c=X_test.shape
	# X_test=X_test.reshape((a,b*c))
	# a,b,c=y_test.shape
	# y_test=y_test.reshape((a,b*c))
	# a,b,c=y_train.shape
	# y_train.shape=((a,b*c))
	# print(y_test)
	lda = LinearDiscriminantAnalysis(n_components=a)
	lda.fit(X_train,y_train)
	X_train=lda.transform(X_train)
	X_test=lda.transform(X_test)
	image_amount = len(X_test)
	feature_amount = len(X_test[0])
	print("images {}, features {}".format(
          image_amount, feature_amount))
	#X_train, X_test, y_train, y_test = pre_process(X, y)
	#SVM_recommend_run("PCA", X_train, X_test, y_train, y_test, {'pca':a})
	SVM_paras = {'C': 0.01, 'max_iter': 2000}
	clf=LinearSVC(**SVM_paras)
	print("start training")
	clf.fit(X_train,y_train)
	print("start testing")
	y_pred=clf.predict(X_test)
	sc = clf.score(X_test, y_test)
	f1_sc = f1_score(y_test, y_pred, average='macro')
	print("score is {}, f1_score is {}".format(sc, f1_sc))
	f=open("lda_result.txt",'a')
	f.write("{} {} {}".format(a,sc,f1_sc))
	f.write('\n')
	f.close()
	



if __name__ == "__main__":
	X, y = load_data()
	X_train, X_test, y_train, y_test = pre_process(X, y)
	for a in range(2,128,2):
		apply_lda(a,X_train, X_test, y_train, y_test)
	#apply_lda(1024,X,y)
	#apply_lda(512,X,y)
	#apply_lda(256,X,y)
	#apply_lda(128,X,y)
	#apply_lda(8,X,y)
	
