import sys
sys.path.append('..')

from boruta import Boruta
import os

from configs import *
from baseline import SVM_recommend, SVM_recommend_run
from load_data import load_data
from pre_process import pre_process

if __name__ == "__main__":
    os.chdir('..')
    clf = SVM_recommend()
    feat_selector = BorutaPy(clf, n_estimators='auto')
    feat_selector.fit(X,y)
    print(feat_selector.support_)
    selected = X[:,feat_selector.support_]
    print ("")
    print ("Selected Feature Matrix Shape")
    print (selected.shape)
    X_train, X_test, y_train, y_test = pre_process(X, y)
    SVM_recommend_run(BORUTA, X_test, X_test, y_train, y_test, {'feature_num': selected.shape[1]})