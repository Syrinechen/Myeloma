#%% imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import PCA
from scipy.linalg import subspace_angles, qr,svd,orth,norm,inv, norm
import Models.process_data as process_data
import Models.models as models

#%% get data
data=process_data.read_data()
X,y=process_data.split_x_y(data)
X=process_data.scale_data(X,'Max')
NEG,POS=process_data.get_groups(pd.concat([X,y],axis=1))
NEG=NEG.drop(['MRD Response'],axis=1).T
POS=POS.drop(['MRD Response'],axis=1).T

# %%
def find_principal_vectors(NEG,POS):
    #find orthonormal basis for both matrices 
    Q_n,Q_p=orth(NEG.values),orth(POS.values)
    #SVD
    U,s,Vh=svd(Q_n.T@Q_p)
    
    #Get principal vectors V_ i.e meta-patients from the two subspaces 
    #(n for mrd- and p for mrd+)
    V_n=Q_n@U
    V_p=Q_p@Vh

    #nb_interations
    q=3
    j=0
    fig,axs=plt.subplots(nrows=2,ncols=4,figsize=(20,20))
    for i in [0,234]:
        v_n=V_n[:,i] #current mrd- meta-patient
        v_p=V_p[:,i] #current mrd+ meta-patient

        v_npp=np.zeros(v_n.shape) 
        v_nnn=np.zeros(v_n.shape)
        v_npp[np.where(v_n>0)]=v_n[np.where(v_n>0)] #extract its positive part
        v_nnn[np.where(v_n<=0)]=v_n[np.where(v_n<=0)] #extract its negatve part

        
        v_ppp=np.zeros(v_p.shape)
        v_pnn=np.zeros(v_p.shape)
        v_ppp[np.where(v_p>0)]=v_p[np.where(v_p>0)] #extract its positive part
        v_pnn[np.where(v_p<=0)]=v_p[np.where(v_p<=0)] #extract its negative part
        
        axs[j,0].scatter(x=v_npp,y=v_ppp,c='r')
        axs[j,0].scatter(x=v_nnn,y=v_pnn,c='b')
        axs[j,0].scatter(x=v_npp,y=v_pnn,c='y')
        axs[j,0].scatter(x=v_nnn,y=v_ppp,c='g')

        p=np.sort(v_n*v_p)
        pp=np.sort(v_npp*v_ppp)
        pn=np.sort(v_nnn*v_pnn)
        n=np.sort(v_n**2+v_p**2)
        nn=np.sort(v_nnn**2+v_pnn**2)
        npos=np.sort(v_ppp**2+v_npp**2)
        
        axs[j,1].semilogx(p)
        axs[j,2].semilogx(n)
        axs[j,3].plot(np.sort(p/n))

        j=j+1

    plt.show()

    return V_p,V_n,s
# %%
find_principal_vectors(NEG,POS)
#%%