#%%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torch.utils.data as Data
import process_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import DEanalysis
import importlib
from skorch import NeuralNetClassifier
#%%
importlib.reload(DEanalysis)
importlib.reload(process_data)
#%%

class MyDataset():
 
  def __init__(self,x,y):

    self.x_train=torch.tensor(x,dtype=torch.float32)
    self.y_train=torch.tensor(y,dtype=torch.float32)
 
  def __len__(self):
    return len(self.y_train)
   
  def __getitem__(self,idx):
    return self.x_train[idx],self.y_train[idx]

class clf(nn.Module):
    
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(clf, self).__init__()
        self.fc1=nn.Linear(input_dim,hidden_dim)
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        self.fc2=nn.Linear(hidden_dim,output_dim)
        self.fc3=nn.LogSoftmax(dim=1)
        torch.nn.init.xavier_uniform(self.fc2.weight)
        
    def forward(self,x):
        return self.fc3(self.fc2(F.relu(self.fc1(x))))
    
def train_model(model,loss_fn,data_loader=None,epochs=30,optimizer=None):
    size=len(data_loader)/data_loader.batch_size
    for epoch in range(epochs):
        running_loss = 0.0
        running_corrects = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.view([-1, n_features])
            optimizer.zero_grad()
            output = model(x)
            #print(output)
            classes=y.type(torch.int64)
            loss = loss_fn(output, classes)
            loss.backward()
            optimizer.step()
            _,preds = torch.max(output.data,1)
            #print(len(preds))
            # statistics
            running_loss += loss.data.item()
            running_corrects += torch.sum(preds == classes.data)
            #print(running_corrects)
        epoch_loss = running_loss / len(data_loader)
        epoch_acc = running_corrects.data.item() / len(data_loader)
        print('Epoch :{:.4f} Loss: {:.4f} Acc: {:.4f}'.format(
                epoch, epoch_loss, epoch_acc))
    return model

#%%
def test_model(test_loader,model):
    accuracy=0
    for batch_idx, (x, y) in enumerate(test_loader):
        x = x.view([-1, n_features])
        output=model(x)
        _,pred=torch.max(output.data,1)
        accuracy+=torch.sum(pred == y.data)
    return accuracy/len(test_loader)
    
#%%
if __name__=='__main__':
    X,y=process_data.main('/home/irit/Documents/Myeloma/raw_count_mrd_response.csv',0,0,0,'None')
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=10)
    #Select genes with DE analysis
    dds_train , volcano=DEanalysis.DE(X_train,y_train)
    selection=list(volcano['gene']
        [(volcano['padj'] < 0.05) & (np.abs(volcano['log2FoldChange'].values) > 1)])
    #%%
    #scale data with max norm
    X_train=process_data.scale_data(X_train,'Max')
    X_test=process_data.scale_data(X_test,'Max')
    #create trainset and test set
    train_set=MyDataset(X_train[selection].values,y_train.values.astype(int))
    test_set=MyDataset(X_test[selection].values,y_test.values.astype(int))
    n_samples=len(X_train)
    #create dataloaders
    train_loader=torch.utils.data.DataLoader(train_set,batch_size=1,shuffle=False)
    test_loader=torch.utils.data.DataLoader(test_set,batch_size=1,shuffle=False)
    #classifier
    n_features=len(selection)
    #%%
    optimizer = optim.Adam(clf.parameters(),lr=0.0001)
    loss_fn = torch.nn.NLLLoss()
    clf=NeuralNetClassifier(module=clf,criterion=loss_fn,optimizer=optimizer)
    
    #%%
    #train_model(clf,loss_fn,train_loader,30,optimizer)
    #%%
    test_model(test_loader,clf)
# %%
