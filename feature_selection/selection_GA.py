import sys
sys.path.append('..')

import os
import numpy as np
from lib.genetic import *
from configs import *
from load_data import load_data_small
from pre_process import pre_process
from baseline import SVM_recommend, SVM_recommend_run

if __name__ == "__main__":
    os.chdir('..')
    clf = SVM_recommend()
    X, y = load_data_small()
    for i in range(1,10):
        ga_selector = FeatureSelectionGA(clf,X,y,verbose=1)
        feature_num = 200*i
        pop = ga_selector.generate(feature_num)
        X_ = X[:,pop]
        X_train, X_test, y_train, y_test = pre_process(X_, y,bReset=True)
        SVM_recommend_run(GA, X_train, X_test, y_train, y_test, {'feature_num': feature_num})
