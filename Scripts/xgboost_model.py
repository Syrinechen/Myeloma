#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import cv

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score,roc_curve, f1_score, RocCurveDisplay, auc,confusion_matrix
from tqdm import trange, tqdm
import seaborn as sns

import process_data
import DEanalysis
import importlib

from sklearn.model_selection import GridSearchCV
from joblib import parallel_backend

importlib.reload(DEanalysis)
#%% n_splits Kfold to evaluate model performance
def xgb_classification(X,y,n_splits):
    metrics_test = {
        'Accuracy': np.zeros(n_splits),
        'AUC': np.zeros(n_splits),
        }
    metrics_train = {
        'Accuracy': np.zeros(n_splits),
        'AUC': np.zeros(n_splits),
        }
    skf = StratifiedKFold(n_splits=n_splits,random_state=3,shuffle=True)
    i=0
    for train, test in skf.split(X, y):
        X_train=X.iloc[train]
        X_test=X.iloc[test]
        y_train=y.iloc[train]
        y_test=y.iloc[test]   
        clf=find_best_params(X_train,y_train,3)
        y_pred=clf.predict(X_test)
        metrics_train['Accuracy'][i]=accuracy_score(y_train,clf.predict(X_train))
        metrics_train['AUC'][i]=accuracy_score(y_train,clf.predict(X_train))
        metrics_test['Accuracy'][i]=accuracy_score(y_test,y_pred)
        metrics_test['AUC'][i]=roc_auc_score(y_test,y_pred)  
        i=i+1  
    return metrics_train,metrics_test
#%% trains model with grid search to find optimal params and return trained model
def find_best_params(X_train,y_train,n_cv):
    # define data_dmatrix
    #data_dmatrix = xgb.DMatrix(data=X,label=y,nthread=-1)    
    # declare parameters
    params = {
                'objective':['binary:logistic'],
                'gamma':[0,10],
                'max_depth': [6,8,10],
                'lambda': [0.001,0.01,0,10,100],
                'alpha':[0,10,0.1],
                'learning_rate':[0.001,0.01,0.1],
                'n_estimators':[5,8,10],   
                'subsample':[0.8,1],  
            }         
    # instantiate the classifier 
    xgb_clf = XGBClassifier(nthread=-1)
    #num_round = 10
    clf = GridSearchCV(xgb_clf, params, verbose=0,
                       n_jobs=-1,cv=n_cv)
    #parallel grid search
    with parallel_backend('threading'):
        clf.fit(X_train,y_train)          
    return clf

#%% Function pour équilibrer le dataset (utile quand on utilise le dataset avec réponses post induction, il n'est pas équilibré)
def undersample(X, y, ratio=1.0):
    """
    Sous-échantillonne les classes majoritaires pour équilibrer le dataset.

    Paramètres :
    X : numpy.array, shape (n_samples, n_features)
        Matrice de données.
    y : numpy.array, shape (n_samples,)
        Vecteur des labels.
    ratio : float (0 <= ratio <= 1), (facultatif, par défaut = 1.0)
        Ratio de sous-échantillonnage pour les classes majoritaires.

    Retourne :
    X_resampled : numpy.array, shape (n_samples_resampled, n_features)
        Matrice de données sous-échantillonnée.
    y_resampled : numpy.array, shape (n_samples_resampled,)
        Vecteur des labels sous-échantillonné.
    """
    # Compter le nombre d'instances pour chaque classe
    class_counts = Counter(y)
    majority_class = max(class_counts, key=class_counts.get)
    minority_class = min(class_counts, key=class_counts.get)
    # Déterminer le nombre d'instances à garder pour la classe majoritaire
    n_samples_to_keep = int(ratio * class_counts[minority_class])
    # Indices des instances de la classe majoritaire
    majority_indices = y.index[y == majority_class]
    # Indices aléatoires à conserver pour la classe majoritaire
    keep_indices = np.random.choice(majority_indices, n_samples_to_keep, replace=False)
    # Indices des instances de la classe minoritaire
    minority_indices = y.index[y == minority_class]
    # Concaténer les indices des classes majoritaires et minoritaires sélectionnées
    selected_indices = np.concatenate([minority_indices, keep_indices])
    # Sous-échantillonnage de X et y
    X_resampled = X.loc[selected_indices]
    y_resampled = y.loc[selected_indices]
    return X_resampled, y_resampled
# %%
if __name__== "__main__":
    csv_file = '/home/irit/Documents/Myeloma_Syrine/Data/TPM_count_mrd_response.csv'
    X, y = process_data.main(csv_file, 0, 0, 10, 'Max')
    res_train,res_test=xgb_classification(X,y,5)
# %%
