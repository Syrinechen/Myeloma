# %% imports
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import normalize, StandardScaler, MaxAbsScaler
from sklearn.feature_selection import VarianceThreshold
#%% function to filter according to var threshold
def filter_variance(X,threshold):
    sel=VarianceThreshold(threshold)
    return pd.DataFrame(data=sel.fit_transform(X),index=X.index, columns=sel.get_feature_names_out(None))
def filter_read_count(X,threshold):
    sum=X.sum(axis=0)
    sum.head()
    return X[X.columns[sum>threshold]]
# %%
def scale_data(X,scaling):
    if (scaling=='Standard'):
        X_scaled=StandardScaler().fit_transform(X)
    elif (scaling=='Max'):
        X_scaled=MaxAbsScaler().fit_transform(X)
    return pd.DataFrame(data=X_scaled,index=X.index,columns=X.columns)

#parameter classif : enter 0 if it's classification, 1 if it's regression
#parameter scaling : 'None' if no scaling, other possibilities: 'Max' or 'Standard'
def main(csv_file,classif,var_threshold,count_threshold,scaling):
    data=pd.read_csv(csv_file,index_col='patient_id')
    if (classif==0):
        y=data['MRD Response']
        X=data.drop(['MRD Response'],axis=1)
    else:
        y=(data['MRD Response'],data['MRD Rate'])
        X=data.drop(['MRD Rate','MRD Response'],axis=1)
    #X=filter_variance(X,var_threshold)
    if (count_threshold>0):
        X=filter_read_count(X,count_threshold)
    if scaling!='None':
        X=scale_data(X,scaling)
    return X,y
# %%
