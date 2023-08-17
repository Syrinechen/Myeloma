#%%
import sys
import pandas as pd
import numpy as np
from stabl.stabl import Stabl, plot_stabl_path, plot_fdr_graph, export_stabl_to_csv, save_stabl_results
from stabl.single_omic_pipelines import single_omic_stabl, single_omic_stabl_cv
from stabl.preprocessing import LowInfoFilter
from stabl.visualization import boxplot_features, scatterplot_features

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split,RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import clone

from sklearn.metrics import accuracy_score

from sklearn.linear_model import Lasso, LogisticRegression, LogisticRegressionCV

import process_data
from joblib import parallel_backend
#%%
file='/home/irit/Documents/Myeloma/TPM_count_post_induction_response.csv'
X,y=process_data.main(file,0,0,10,'Max')
#%%
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y)
# %%
logit_lasso = LogisticRegression(penalty="l1", max_iter=int(1e6), solver="liblinear", class_weight="balanced")
# %%
stabl_class = Stabl(
    base_estimator=clone(logit_lasso),
    lambda_name="C",
    lambda_grid=np.logspace(0,4,10),
    n_bootstraps=100,
    random_state=42,
    artificial_type='random_permutation',
    verbose=4,
)
# %%
with parallel_backend('threading'):
    stabl_class.fit(X_train,y_train)
# %%
save_stabl_results(stabl_class,'/home/irit/Documents/Myeloma/Models/stabl_results',X_train,y_train)
#%%
plot_stabl_path(stabl_class)
#%%
plot_fdr_graph(stabl_class)
#%%
selected_genes=stabl_class.get_feature_names_out()
# %%
boxplot_features(stabl_class.get_feature_names_out(), X_train, y_train)
# %%
clf=LogisticRegressionCV(Cs=np.logspace(0,2,100))
clf.fit(X_train[selected_genes],y_train)
preds=clf.predict(X_test[selected_genes])
# %% Test 
stabl = Stabl(
    lambda_grid=np.linspace(0.01, 5, 10),
    n_bootstraps=1000,
    artificial_type="random_permutation",
    replace=False,
    fdr_threshold_range=np.arange(0.1, 0.9, 0.01),
    sample_fraction=.8,
    random_state=42,
)
stability_selection = clone(stabl).set_params(hard_threshold=.1, artificial_type = None)
outer_splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
predictions_dict = single_omic_stabl_cv(
    X=X_train,
    y=y_train,
    outer_splitter=outer_splitter,
    stabl=stabl,
    stability_selection=stability_selection,
    task_type="binary",
    save_path='/home/irit/Documents/Myeloma/Models/stabl_ss_results',
    outer_groups=None)
# %%
