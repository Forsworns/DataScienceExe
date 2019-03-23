#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'feature_selection\lib'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# ### Using Boruta on the Madalon Data Set
# Author: [Mike Bernico](mike.bernico@gmail.com)
# 
# This example demonstrates using Boruta to find all relevant features in the Madalon dataset, which is an artificial dataset used in NIPS2003 and cited in the [Boruta paper](https://www.jstatsoft.org/article/view/v036i11/v36i11.pdf)
# 
# This dataset has 2000 observations and 500 features.  We will use Boruta to identify the features that are relevant to the classification task.
# 
# 
# 
# 

#%%
# Installation
#!pip install boruta


#%%
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy


#%%
def load_data():
    # URLS for dataset via UCI
    train_data_url='https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.data'
    train_label_url='https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.labels'
        
    
    X_data = pd.read_csv(train_data_url, sep=" ", header=None)
    y_data = pd.read_csv(train_label_url, sep=" ", header=None)
    data = X_data.ix[:,0:499]
    data['target'] = y_data[0] 
    return data


#%%
data = load_data()


#%%
data.head()


#%%
y=data.pop('target')
X=data.copy()

#%% [markdown]
# Boruta conforms to the sklearn api and can be used in a Pipeline as well as on it's own. Here we will demonstrate stand alone operation.
# 
# First we will instantiate an estimator that Boruta will use.  Then we will instantiate a Boruta Object.

#%%
rf = RandomForestClassifier(n_jobs=-1, class_weight='auto', max_depth=7)
# define Boruta feature selection method
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2)

#%% [markdown]
# Once built, we can use this object to identify the relevant features in our dataset.

#%%
feat_selector.fit(X,y)

#%% [markdown]
# Boruta has confirmed only a few features as useful.   When our run ended, Boruta was undecided on 2 features.   '
# 
# We can interrogate .support_ to understand which features were selected.   .support_ returns an array of booleans that we can use to slice our feature matrix to include only relevant columns.   Of course, .transform can also be used, as expected in the scikit API.

#%%
# check selected features
print(feat_selector.support_)
#select the chosen features from our dataframe.
selected = X.ix[:,feat_selector.support_]
print ("")
print ("Selected Feature Matrix Shape")
print (selected.shape)

#%% [markdown]
# We can also interrogate the ranking of the unselected features with .ranking_

#%%
feat_selector.ranking_


#%%



