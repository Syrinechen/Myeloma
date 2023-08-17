# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.spatial import distance
from scipy.cluster import hierarchy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from scipy.cluster.hierarchy import fcluster
from scipy.spatial import distance
from scipy.cluster import hierarchy
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
import process_data

def plot_statistics(X,stats):
    plt.figure()
    plt.title('Somme des comptes par gène')
    sum=X.sum(axis=0)
    plt.loglog(sum.sort_values().values)
    plt.figure()
    plt.title('Expression moyenne par gène')
    plt.loglog(stats.T['mean'].sort_values().values)
    plt.show()
    plt.figure()
    plt.title('Ecart type par gènes')
    plt.semilogy(stats.T['std'].sort_values().values)
    plt.show()
    CV=stats.T['std']/stats.T['mean']
    plt.figure()
    plt.title('Coefficient de variation par gène')
    plt.semilogy(CV.sort_values().values)
    plt.show()
    return stats

def patients_corr_analysis(X, scaling, dist):
    if (scaling == 'Standard'):
        X = StandardScaler().fit_transform(X)
    elif (scaling == 'Max'):
        X = MaxAbsScaler().fit_transform(X)
    CorrPatients = np.corrcoef(X)
    # cluster patients according to correlations

    row_linkage = hierarchy.linkage(
        distance.pdist(CorrPatients,metric=dist), method='complete')
    col_linkage = hierarchy.linkage(
        distance.pdist(CorrPatients.T,metric=dist), method='complete')
    cl = sns.clustermap(CorrPatients, row_linkage=row_linkage,
                        col_linkage=col_linkage, figsize=(20, 20), cmap="YlGnBu")
    return row_linkage

# What is inside the clusters ?
def get_clusters(X, linkage, nb_clusters):
    fl = fcluster(linkage, nb_clusters, criterion='maxclust')
    clusters = []
    for i in range(nb_clusters):
        df = X.iloc[np.where(fl == i+1)]
        clusters.append(df)
    return clusters

def get_gene_stats(cluster):
    return cluster.describe()

def get_MRD_stats(cluster, y):
    return pd.concat([cluster, y], axis=1).groupby('MRD Response').count().iloc[:, 1][0]/len(cluster)

def patients_clusters_analysis(X, y, linkage, nb_clusters):
    clusters = get_clusters(X, linkage, nb_clusters)
    for i in range(len(clusters)):
        print('Le pourcentage de patients qui sont détectés MRD- est : ', get_MRD_stats(
            clusters[i], y), ',le nombre de patients dans le cluster est : ', len(clusters[i]))
    return clusters

def main(X, y, scaling, nb_clusters,dist):
    linkage = patients_corr_analysis(X, scaling,dist)
    clusters = patients_clusters_analysis(X, y, linkage, nb_clusters)
    return clusters

# %%


if __name__ == "__main__":
    csv_file='/home/irit/Documents/Myeloma_Syrine/Data/raw_count_mrd_response.csv'
    X,y=process_data.main(csv_file,0,0,0,'None')
    linkage = patients_corr_analysis(X,'Max','euclidean')
    patients_clusters_analysis(X, y, linkage, nb_clusters=2)

    # %%
