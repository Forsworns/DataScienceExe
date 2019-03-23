import sys
sys.path.append('..')

import os

from genetic import *
from configs import *
from load_data import load_data
from pre_process import pre_process
from baseline import SVM_recommend, SVM_recommend_run

if __name__ == "__main__":
    os.chdir('..')
    clf = SVM_recommend()
    X, y = load_data()
    ga_selector = FeatureSelectionGA(clf,X,y,verbose=1)
    feature_num = 200
    pop = fsga.generate(feature_num)
    X = X[:,pop]
    X_train, X_test, y_train, y_test = pre_process(X, y)
    SVM_recommend_run(GA, X_train, X_test, y_train, y_test, {'feature_num': feature_num})
