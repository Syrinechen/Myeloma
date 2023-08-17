#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score,roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import NMF
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV,RidgeClassifierCV
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import process_data as process_data
import feature_selection
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix

#%%
def linear_model(X,y,model,n_splits):
    metrics_test = {
        'Accuracy': np.zeros(n_splits),
        'AUC': np.zeros(n_splits),
        'f1 score':np.zeros(n_splits),
        }
    metrics_train = {
        'Accuracy': np.zeros(n_splits),
        'AUC': np.zeros(n_splits),
        'f1 score':np.zeros(n_splits),
        }
    skf = StratifiedKFold(n_splits=n_splits,random_state=3,shuffle=True)
    i=0
    for train, test in skf.split(X, y):
        clf=model
        X_train=X.iloc[train]
        X_test=X.iloc[test]
        y_train=y.iloc[train]
        y_test=y.iloc[test]   
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        metrics_train['Accuracy'][i]=accuracy_score(y_train,clf.predict(X_train))
        metrics_train['AUC'][i]=accuracy_score(y_train,clf.predict(X_train))
        metrics_test['Accuracy'][i]=accuracy_score(y_test,y_pred)
        metrics_test['AUC'][i]=roc_auc_score(y_test,y_pred)  
        i=i+1  
    return metrics_train,metrics_test

#%%
if __name__=="__main__":
    csv_file = '/home/irit/Documents/Myeloma/TPM_count_mrd_response.csv'
    X, y = process_data.main(csv_file, 0, 0, 10, 'Max')
    #%%
    results_train,results_test=linear_model(X,y,LogisticRegression(max_iter=int(1e8)),5)
    
# %%
