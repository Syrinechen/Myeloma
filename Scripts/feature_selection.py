#%%
import pandas as pd
import numpy as np
import random as rd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV,LinearRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from boruta import BorutaPy

import process_data 


def boruta(n_trials,X_train,y_train):
    ###initialize Boruta
    forest = XGBClassifier(
    n_jobs = -1, 
    n_estimators=1000,
    )
    
    boruta = BorutaPy(
    estimator = forest, 
    n_estimators = 1000,
    max_iter = n_trials, # number of trials to perform
    verbose=2,
    )   
    ### fit Boruta (it accepts np.array, not pd.DataFrame)
    boruta.fit(np.array(X_train.values), np.array(y_train.values))

    ### print results
    green_area = X_train.columns[boruta.support_].to_list()
    blue_area = X_train.columns[boruta.support_weak_].to_list()
    print('features in the green area:', green_area)
    print('features in the blue area:', blue_area)
    return green_area+blue_area
#%%

def lasso (X_train,y_train):
    model=LogisticRegressionCV(penalty='l1',solver='liblinear',Cs=np.logspace(-2,2,100),cv=3)
    feature_names=np.array(X_train.columns).flatten()
    model=model.fit(X_train,y_train)
    print(model.C_)
    importance=np.abs(model.coef_.flatten())
    feature_names=feature_names[np.argsort(importance)]
    importance=np.sort(importance)
    selected_coefs = pd.DataFrame(
    importance[importance>0],
    columns=["Coefficients importance"],
    index=feature_names[importance>0],
    )
    #print(importance)
    #selected_coefs.plot.barh(figsize=(20, 10))
    #plt.title("lasso model")
    #plt.xlabel("Raw coefficient values")
    #plt.axvline(x=0, color=".5")
    #plt.subplots_adjust(left=0.3)
    selected_genes=list(selected_coefs.index)
    #plot validation curve 
    """params=model.Cs_
    scores=model.scores_[1].mean(axis=0)
    fig,ax=plt.subplots(figsize=(5,5))
    ax.plot(params,scores)
    ax.set_xscale('log')
    ax.set_xlabel('C')
    ax.set_ylabel('Test accuracy')
    plt.set_title('Validation curve')"""
    return selected_genes,model


# %%
if __name__=='__main__':
    path='/home/irit/Documents/Myeloma/TPM_count_mrd_response.csv'
    X,y=process_data.main(path,0,0,10,'Max') 
    #%%
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,stratify=y)
    # %%
    selected,model=boruta(100,X_train,y_train)
# %%
