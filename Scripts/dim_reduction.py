#%% imports
import pandas as pd
import numpy as np
import random as rd
import seaborn as sns
from sklearn.decomposition import NMF, PCA
from random import random
import matplotlib.pyplot as plt
import process_data
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from scipy.spatial import distance
from scipy.cluster import hierarchy
from tqdm import trange
from sklearn.preprocessing import StandardScaler
#%%NMF
class nmf_model():
    def __init__(self,X,y) :
        self.data=X
        self.classes=y
        self.n_samples=X.shape[1]
        self.n_genes=X.shape[0]
        self.n_runs=1
        self.A_k=None
        self.best_k=None
        self.delta_k=None
        self.best_k=None
        self.model=None
        self.W=None
        self.H=None
        
    
    #Returns consensus matrix, which is the averaged connectivity matrix on several runs of the algorithm
    def get_consensus_matrix(self,n_components):
        model=NMF(n_components,init='random')
        M_k=np.zeros((self.n_samples,self.n_samples))
        print('start runs')
        for i in trange(self.n_runs):
            W=model.fit_transform(self.data)
            H=model.components_
            n_metagenes=H.shape[0]

            #calculate Connectivity matrix
            clusters=np.zeros(self.n_samples)
            C=np.zeros((self.n_samples,self.n_samples))

            for i in range (self.n_samples):
                clusters[i]=np.argmax(H[:,i])

            for i in range (self.n_samples):
                for j in range (i,self.n_samples):
                    if (clusters[i]==clusters[j]):
                        C[i,j]=1
                    else:
                        C[i,j]=0
        
            M_k=M_k+C

        M_k=M_k/self.n_runs
        #consensus matrix heatmap
        row_linkage = hierarchy.linkage(
            1-M_k, method='complete')
        col_linkage = hierarchy.linkage(
            1-M_k, method='complete')
        cl=sns.clustermap(M_k,row_linkage=row_linkage,col_linkage=col_linkage,figsize=(5, 5), cmap="YlGnBu")
        neg=np.where(y==0)[0].tolist()
        pos=np.where(y==1)[0].tolist()
        m_k_reordered=np.concatenate((M_k[pos,:],M_k[neg,:]),axis=0)
        #sns.heatmap(m_k_reordered)
        return M_k
        
    #To evaluate model stability
    def get_consensus_distribution (self,M_k):
        list_entries=M_k.ravel()
        print(max(list_entries))
        hist, bins=np.histogram(list_entries, density=True)
        plt.figure(figsize=(5,5))
        plt.hist(list_entries,bins=np.linspace(0,1,10))
        plt.show()
        #calculate CDF
        cdf=np.cumsum(hist)
        return bins, hist, cdf

    def model_selection(self):
        list_k=[100,1000,10000]
        self.A_k=np.zeros(len(list_k))
        for i in trange(len(list_k)):
            print('k= ',list_k[i])
            Ck=self.get_consensus_matrix(list_k[i])
            bins,hist,cdf=self.get_consensus_distribution(Ck)
            self.A_k[i] = np.sum(h*(b-a) for b,a,h in zip(bins[1:],bins[:-1],cdf))
        #differences between areas under CDFs
        actual=0
        self.delta_k=(np.zeros(len(list_k)))
        for i in range (len(list_k)):
            self.delta_k[i]=self.A_k[i]-actual
            actual=self.A_k[i]
        print('area under cdf: ',self.A_k)
        print('difference between area under cdf',self.delta_k)
        plt.plot(list_k,self.delta_k)
        self.best_k=list_k[np.argmax(self.delta_k)]
    
    def build_best_model(self):
        self.model_selection()
        self.model=NMF(n_components=self.best_k)
        self.W=self.model.fit_transform(self.data)
        self.H=self.model.components_
        print(self.H)

#%% pca
def get_nb_components(X, threshold):
    max_components=min(X.shape[0],X.shape[1])
    scaler = StandardScaler()
    scaler.fit_transform(X)
    pca = PCA(n_components=max_components)
    pca.fit(X)
    explained_var=pca.explained_variance_ratio_.cumsum()
    
    #plot explained variance
    fig, ax = plt.subplots(figsize=[10,5])
    xi =np.arange(1, max_components+1, step=1)
    plt.ylim(0.0,1.1)
    plt.plot(xi, explained_var, marker='o', linestyle='-', color='black')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative variance (%)')
    plt.title('The number of components needed to explain variance')
    plt.axhline(y=threshold, color='grey', linestyle='--')
    plt.text(1.1, 1, '95% cut-off threshold', color = 'black', fontsize=16)
    ax.grid(axis='x')
    plt.tight_layout()
    plt.savefig('pcavisualize_1.png', dpi=300)
    plt.show()
    return min(np.where(explained_var>0.95)[0])+1 

def pca_transform(X,n_components):
    scaler = StandardScaler()
    scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    pca.fit(X)
    X_pca=pca.transform(X)
    return X_pca,pca
#%%
if __name__=='__main__':
    X,y=process_data.main('/home/irit/Documents/Myeloma/TPM_count_mrd_response.csv',0,0,5,'Max')
    build_model=True
    if (build_model):
        model=nmf_model(X.T,y)
        model.build_best_model()
   