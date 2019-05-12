# load data of features and labels
import numpy as np
from configs import *
import pandas as pd

def load_data(dataset=0):
    '''
    :param dataset: the order of the dataset, 0 for problem a, 1 for b, 2 for c
    '''
    data_src = pd.read_csv(SRC_FILE[dataset])
    data_tgt = pd.read_csv(TGT_FILE[dataset]) 
    data_src = np.array(data_src)   
    data_tgt = np.array(data_tgt)   
    X_src = data_src[:,0:-1]
    y_src = data_src[:,-1]
    X_tgt = data_tgt[:,0:-1]
    y_tgt = data_tgt[:,-1]
    return [X_src, y_src, X_tgt, y_tgt]
    

if __name__ == "__main__":
    X_src, y_src, X_tgt, y_tgt = load_data(dataset=1)
    image_amount = len(X_src)
    feature_amount = len(X_src[0])
    label_amount = len(y_src)
    type_amount = len(set(y_src))
    print("images {}, features {}, labels {}, types {}".format(
          image_amount, feature_amount, label_amount, type_amount))
