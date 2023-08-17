#%%
import pandas as pd
import numpy as np
import umap
import process_data
import matplotlib.pyplot as plt
import seaborn as sns

#%% download dataset from git https://github.com/clinicalml/ml_mmrf and put path in csv_file variable
csv_file='/home/irit/Documents/Myeloma/commpass dataset/MMRF_CoMMpass_IA15a_E74GTF_Cufflinks_Gene_FPKM.txt'
def process_data_(csv_file,old_data):
    #read file 
    new_data=pd.read_csv(csv_file,sep='\t').T
    #change header
    new_genes=new_data.iloc[0,:].values.tolist()
    new_data.columns=new_genes
    new_data=new_data.drop(['GENE_ID','Location'],axis=0)
    #get our datasets' genes
    old_genes=old_data.columns.tolist()
    #change normalization to TPM
    new_data=FPKM_to_TPM(new_data)
    new_data=new_data[old_genes]
    return new_data.loc[:,~new_data.columns.duplicated()]
    
#%%
def FPKM_to_TPM(df):
    norma_factors=df.sum(axis=1)
    for i in range(len(df)):
        if norma_factors[i]!=0:
            df.iloc[i,:]=(df.iloc[i,:]/norma_factors[i])*1e6
    return df    

def main():
    csv_file='/home/irit/Documents/Myeloma/commpass dataset/MMRF_CoMMpass_IA15a_E74GTF_Cufflinks_Gene_FPKM.txt'
    X_old,y_old=process_data.main('/home/irit/Documents/Myeloma/TPM_count_mrd_response.csv',0,0,0,'None')
    data=process_data_(csv_file,X_old)
    return pd.concat([X_old,data],axis=0)   
#%%
if __name__=='__main__':
    csv_file='/home/irit/Documents/Myeloma/commpass dataset/MMRF_CoMMpass_IA15a_E74GTF_Cufflinks_Gene_FPKM.txt'
    X_old,y_old=process_data.main('/home/irit/Documents/Myeloma/TPM_count_mrd_response.csv',0,0,0,'None')
    new_data=process_data_(csv_file,X_old)
    #%%
    dataset=pd.concat([X_old,new_data],axis=0)  
    dataset.to_csv('/home/irit/Documents/Myeloma/TPM_count_augmented.csv')
    
    #%%
    dataset=process_data.scale_data(dataset,'Standard')
    X_old=process_data.scale_data(X_old,'Max')
    new_data=process_data.scale_data(new_data,'Max')
    #%%
    reducer = umap.UMAP(metric='cosine')
    embedding = reducer.fit_transform(dataset)
    # %%
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=['r' if patient in X_old.index else 'b' for patient in dataset.index]
        )
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of the MMyeloma dataset', fontsize=24)
    #%%
