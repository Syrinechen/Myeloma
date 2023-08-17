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
import xgboost_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV,LogisticRegression
from sklearn.metrics import accuracy_score
import importlib
from tqdm import tqdm, trange

#%%reload
importlib.reload(xgboost_model)
#%%

class MyDataset():
 
  def __init__(self,x):

    self.x=torch.tensor(x,dtype=torch.float32)
 
  def __len__(self):
    return len(self.x)
   
  def __getitem__(self,idx):
    return self.x[idx]
  
#%%
class AE(nn.Module):
    def __init__(self, input_dim,out_dim):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=input_dim, out_features=out_dim
        )
        self.encoder_output_layer = nn.Linear(
            in_features=out_dim, out_features=out_dim
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=out_dim, out_features=out_dim
        )
        self.decoder_output_layer = nn.Linear(
            in_features=out_dim, out_features=input_dim)

    def forward(self, features,return_encoding=False):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        if return_encoding:
            return code,reconstructed
        return reconstructed

#%%

# VAE model
class VAE(nn.Module):
    def __init__(self, input_dim, h_dim, z_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, input_dim)
        
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc4(z))
        return torch.sigmoid(self.fc5(h))
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var

#%%
def train_model(model,loss_fn,data_loader,epochs,optimizer):
    for epoch in trange(epochs):
        loss=0
        for batch_idx, data in tqdm(enumerate(data_loader)):
            data = data.view([-1, n_features])
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, data)
            loss.backward()
            optimizer.step()
            # add the mini-batch training loss to epoch loss
            loss += loss.item()
        # compute the epoch training loss
        loss = loss / len(data_loader)
        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
#%%
def train_model_variational(model, loss_fn,data_loader,num_epochs,optimizer):
    # Start training
    for epoch in range(num_epochs):
        for i, (x, _) in enumerate(data_loader):
            # Forward pass
            x = x.view(-1, input_dim)
            x_reconst, mu, log_var = model(x)
            # Compute reconstruction loss and kl divergence
            # For KL divergence between Gaussians, see Appendix B in VAE paper or (Doersch, 2016):
            # https://arxiv.org/abs/1606.05908
            reconst_loss = loss_fn(x,x_reconst)
            kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            # Backprop and optimize
            loss = reconst_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 10 == 0:
                print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}" 
                        .format(epoch+1, num_epochs, i+1, len(train_loader), reconst_loss.item()/len(x), kl_div.item()/len(x)))
        
#%%
def train_pred_model(ae,prediction_model,labelled_data,test):
    X=process_data.scale_data(labelled_data[0],'Max')
    y=labelled_data[1]
    X_test=ae.forward(torch.tensor(X.iloc[test].values,dtype=torch.float32),return_encoding=True)[0]
    print(X_test.shape)
    y_test=y.iloc[test]
    train=[i for i in range (len(X)) if i not in list(test)]
    X_train=ae.forward(torch.tensor(X.iloc[train].values,dtype=torch.float32),return_encoding=True)[0]
    y_train=y.iloc[train]
    print(X_train.shape)
    prediction_model.fit(X_train.detach().numpy(),y_train)
    preds=prediction_model.predict(X_test.detach().numpy())
    return accuracy_score(y_test,preds)
    
#%%

def test_model(model,prediction_model,test_loader):
    res=[]
    # loop, over whole test set
    for i, batch in enumerate(test_loader):
        embedding,output=model.forward(batch,return_encoding=True)
        res.append(prediction_model.predict(output.detach().numpy()))  
    return res
#%%
if (True):
   X=pd.read_csv('/home/irit/Documents/Myeloma/TPM_count_augmented.csv')
   #%%
   labelled=process_data.main('/home/irit/Documents/Myeloma/TPM_count_mrd_response.csv',
                              0,0,0,'Max')
   #%%
   X.index=X['Unnamed: 0']
   X=X.drop(['Unnamed: 0'],axis=1)
   #%%
   #select patients for testing
   test=np.random.randint(0,582,30)
   X_test=X.iloc[test]
   train=[i for i in range (len(X)) if i not in list(test)]
   X_train=process_data.scale_data(X.iloc[train],'Max')
   #%%
   #create dataset
   dataset=MyDataset(X_train.values)
   #%%
   n_samples=len(X_train)
   n_features=X.shape[1]
   #create dataloaders
   loader=torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=True)
   input_dim = n_features
   encoding_dim = 1000
   #hidden_dim=10000
   #create model
   model = AE(input_dim,encoding_dim)
   optimizer = optim.Adam(model.parameters(),lr=0.0001)
   loss_fn = torch.nn.MSELoss()
   train_model(model, loss_fn, data_loader=loader, epochs=10, optimizer=optimizer)
   #%%
   print(train_pred_model(model,LogisticRegression(),labelled,test))
# %%
# Hyper-parameters
input_dim = n_features
h_dim = 1000
z_dim = 100
num_epochs = 40
learning_rate = 1e-3

model = VAE(input_dim,h_dim,z_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
train_model_variational(model,loss_fn,train_loader,10,optimizer)
#%%
prediction_model=RandomForestClassifier(n_estimators=10,random_state=0)
for i,(x,y) in enumerate(train_loader):  
    x_reduced=model.reparameterize(model.encode(x)[0],model.encode(x)[1])
    prediction_model.fit(x_reduced.detach().numpy(),y.detach().numpy())

# %%
y_pred=test_model(model,prediction_model,test_loader)
y_pred=np.concatenate((y_pred[0],y_pred[1]),axis=0)
print(accuracy_score(y_pred,y_test))
# %%
