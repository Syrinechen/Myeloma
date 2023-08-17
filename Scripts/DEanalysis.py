#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydeseq2
from sklearn.model_selection import train_test_split, StratifiedKFold
import pydeseq2
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import process_data
import dash_bio
from sklearn.linear_model import LinearRegression, LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score,roc_curve, f1_score, RocCurveDisplay, auc,confusion_matrix
from scipy.stats import gmean
from statistics import median
from scipy.spatial import distance
from scipy.cluster import hierarchy
from tqdm import trange, tqdm
import seaborn as sns
from scipy.cluster.hierarchy import fcluster
import correlations_analysis
import importlib
import process_data
import dash
from sklearn.model_selection import GridSearchCV
from joblib import parallel_backend
import xgboost_model,SVM_model,Lin_model
# %%
importlib.reload(process_data)
# Median of ratios normalisation method
def MOR_normalization(X, size_factors):
    X_normalized = np.zeros(X.shape)
    for i in trange(len(X)):
        X_normalized[i, :] = X.iloc[i, :].values*size_factors[i]
    return X_normalized
# %%
def DE(X_train, y_train):
    counts_df = X_train
    clinical_df = pd.DataFrame(y_train.astype('int'))
    # single factor analysis
    dds_train = DeseqDataSet(
        counts=counts_df,
        metadata=clinical_df,
        design_factors="MRD Response",
        refit_cooks=True,
        n_cpus=None,
    )
    # apply DESeq model
    dds_train.deseq2()
    # compute statistics
    stat_res = DeseqStats(dds_train, independent_filter=True,
                          cooks_filter=True, n_cpus=None)
    stat_res.summary()
    # plot volcano plot and get differentially expressed gene
    volcano = stat_res.results_df[['log2FoldChange', 'padj']].dropna()
    volcano['gene']=volcano.index
    volcano = volcano.reset_index(drop=True)
    return dds_train,stat_res, volcano
# %%
def DE_analysis_model(X, y,n_splits,subsample_size):
    selected_genes=[]
    volcanos=[]
    #predicted_patients = []
    #skf = StratifiedKFold(n_splits=n_splits)
    for i in range(n_splits):
        X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=1-subsample_size,random_state=i,shuffle=True,stratify=y)
        #predicted_patients.append(list(X.index[val]))
        dds_train,stat_res,volcano = DE(X_train, y_train)
        genes=set(volcano['gene']
        [(volcano['padj'] < 0.05) | (np.abs(volcano['log2FoldChange'].values) > 1)])
        selected_genes.append(genes)
        volcanos.append(volcano)
    intersection=selected_genes[0]
    for j in range(1, len(selected_genes)):
        intersection=intersection&selected_genes[j]        
    return intersection,volcanos
#%%
def cross_val(X,y,resample,cv,n_bootstrap,subsample_size):
    metrics_lr = {
        'Accuracy': np.zeros(cv),
        'AUC': np.zeros(cv)
    }
    metrics_rf = {
        'Accuracy': np.zeros(cv),
        'AUC': np.zeros(cv)
    }
    metrics_xgb={
        'Accuracy': np.zeros(cv),
        'AUC': np.zeros(cv)
    }
    selected_genes=[]
    for i in trange (cv):
        if (resample):
            X_resampled, y_resampled,test_indices = undersample(X, y, ratio=1)
            X_train,X_test,y_train,y_test=train_test_split(X_resampled,y_resampled,test_size=0.1,shuffle=True,stratify=y_resampled)
            X_test=pd.concat([X_test,X.loc[test_indices]],axis=0)
            y_test=pd.concat([y_test,y.loc[test_indices]],axis=0)
        else:
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,shuffle=True,stratify=y)
        if (i==0):
            print(len(X_train),len(X_test),len(y_test))
        #We choose to do 6 splits (computationally too long to select k)
        selected,volcanos=DE_analysis_model(X_train,y_train,n_bootstrap,subsample_size)
        selected_genes.append(selected)
        X_train_scaled=process_data.scale_data(X_train[selected],'Max')
        X_test_scaled=process_data.scale_data(X_test[selected],'Max')
        #clf_1=LogisticRegression(max_iter=int(1e6),solver='lbfgs')
        #clf_1.fit(X_train_scaled,y_train)
        #clf_2=SVM_model.find_best_params(X_train_scaled,y_train,2)
        clf_3=xgboost_model.find_best_params(X_train_scaled,y_train,2)
        #pred_1=clf_1.predict(X_test_scaled)
        #pred_2=clf_2.predict(X_test_scaled)
        pred_3=clf_3.predict(X_test_scaled)
        #metrics_lr['Accuracy'][i]=accuracy_score(y_test,pred_1)
        #metrics_lr['AUC'][i]=roc_auc_score(y_test,pred_1)
        #metrics_rf['Accuracy'][i]=accuracy_score(y_test,pred_2)
        #metrics_rf['AUC'][i]=roc_auc_score(y_test,pred_2)
        metrics_xgb['Accuracy'][i]=accuracy_score(y_test,pred_3)
        metrics_xgb['AUC'][i]=roc_auc_score(y_test,pred_3)
    return metrics_xgb,selected_genes,volcanos
#%%
def select_k(X_train,X_test,y_train,y_test,subsample_size,k_min,k_max):
    genes=[]
    test_acc=[]
    models_=[]
    for i in range(k_min,k_max+1):
        n_splits=i
        selected,volcanos=DE_analysis_model(X_train,y_train,n_splits,subsample_size)
        print('n_splits=',n_splits)
        print('number of genes selected=',len(selected))
        genes.append(selected)
        X_train_scaled=process_data.scale_data(X_train[selected],'Max')
        X_test_scaled=process_data.scale_data(X_test[selected],'Max')
        model=LogisticRegression()
        model=model.fit(X_train_scaled,y_train)
        acc=accuracy_score(model.predict(X_test_scaled),y_test)
        test_acc.append(acc)
        models_.append(model)
        print('acc=',acc)
    return genes,test_acc,models_

# %% for evaluation
def analyze_classifier_performance(y_true, y_pred,model):
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    # Calculate AUC
    auc_score = roc_auc_score(y_true, y_pred)
    # Calculate F1-score
    f1score = f1_score(y_true, y_pred)
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()
    # Create a dictionary to store the metrics
    metrics = {
        'Accuracy': accuracy,
        'AUC': auc_score,
        'F1-score': f1score,
        'ROC Curve': (fpr, tpr, thresholds),
    }
    return metrics
#%% convert selected genes to their actual names (forcheck with doctors)
Coress=pd.read_csv('/home/irit/Documents/Myeloma/Utilities/genes_correspondance_id_name.txt',index_col='Gene_id')
dict_coress={k:v[0] for (k,v) in zip(Coress.index,Coress.values)}
def convert_to_name(list_ids):
    return list( map(dict_coress.get, list_ids) )

def process_de_results(volcanos):
    res={key: value for key,value in []}
    for i in range (len(volcanos)):
        volcano=volcanos[i][0]
        selection=list(volcano['gene']
        [(volcano['pvalue'] < 0.05) & (np.abs(volcano['log2FoldChange'].values) > 1)])
        res[i]=convert_to_name(selection)
    return res

def reorder_patients(X):
    index=list(X.index[y==1])+list(X.index[y==0])
    X_reordered=np.concatenate((X.values[y==1,:],X.values[y==0,:]),axis=0)
    X_reordered=pd.DataFrame(X_reordered,index=index,columns=X.columns)
    return X_reordered

#%% function to equilibrate dataset (because dataset with post_induction labels is highly desiquilibrated ~ 1/5 mrd-, 4/5 mrd+)
from collections import Counter

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
    print(len(selected_indices))
    test_indices=[i for i in y.index if i not in selected_indices ]
    print(len(test_indices))
    print(len(y))
    # Sous-échantillonnage de X et y
    X_resampled = X.loc[selected_indices]
    y_resampled = y.loc[selected_indices]

    return X_resampled, y_resampled,test_indices


#%%
def get_cross_val_inter(selections):
    intersection=[]
    for g in selections[0]:
        if (g in selections[1]) &  (g in selections[2]) & (g in selections[3]) & (g in selections[4]):
            intersection.append(g)
    return intersection
# %%
if __name__ == "__main__":
    csv_file = '/home/irit/Documents/Myeloma/raw_count_post_induction_response.csv'
    X, y = process_data.main(csv_file, 0, 0, 5, 'None')
    #%%
    results_lr,results_rf,results_xgb,selections=cross_val(X,y,True,5)
    #%%
    selected=get_cross_val_inter(selections)
    # %%
    in_both=[]
    for i in range (len(selected_ind)):
        if selected_ind[i] in selected:
            in_both.append(selected_ind[i])