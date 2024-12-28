from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
#%%
# Add the skip connection from the A1 to A3
import torch.nn as nn
class Model(nn.Module):
    def __init__(self,n_0,n_1,n_2,n_3,n_4):
        super().__init__()
        self.L1 = nn.Linear(n_0,n_1)
        self.L2 = nn.Linear(n_1,n_2)
        self.L3 = nn.Linear(n_2,n_3)
        self.L4 = nn.Linear(n_3,n_4)
        self.act1 = nn.ReLU()
        
    def forward(self, X):
        Z1 = self.L1(X)
        A1 = self.act1(Z1)
        Z2 = self.L2(A1)
        A2 = self.act1(Z2)
        Z3 = self.L3(A2)
        A3 = self.act1(Z3)
        ### START CODE HERE ###  (≈ 1 line of code)
        A3 = A3 + A1
        ### END CODE HERE ###
        Z4 = self.L4(A3)
        return Z4
#%%

class MaterialsDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        data = df.to_numpy()
        data = torch.tensor(data,dtype=torch.float32)
        self.Y = data[:,:1]
        self.X = data[:,1:]
        
    def __len__(self):
        ### START CODE HERE ###  (≈ 1 line of code)
        number_of_data = self.Y.shape[0]
        ### END CODE HERE ###
        return number_of_data
    def __getitem__(self, idx):
        ### START CODE HERE ###  (≈ 2 line of code)
        x = self.X[idx,:]
        y = self.Y[idx,:]
        ### END CODE HERE ###
        return x,y
data = MaterialsDataset('HW13.csv')
#%%
# We will split the data in to 90% train, 5% validation, and 5% test.
# It turns out that you can also specify the ratio of the train, validation, 
# instead of the number of data points. 
# Define the following parameters below:
# Rtrain: (float) ratio of training set data
# Rval: (float) ratio of validation set data
# Rtest: (float) ratio of test set data
### START CODE HERE ###  (≈ 3 line of code)
Rtrain = 0.9
Rval = 0.05
Rtest = 0.05
### END CODE HERE ###
data_train, data_val, data_test = random_split(data,[Rtrain,Rval,Rtest])
#%%
# in the code below, make dataloader for training, validation, and test set
# with batch size of 128
### START CODE HERE ###  (≈ 3 line of code)
dataloader_train = DataLoader(data_train, batch_size=128, shuffle=True)
dataloader_val = DataLoader(data_val, batch_size=128, shuffle=True)
dataloader_test = DataLoader(data_test, batch_size=128, shuffle=True)
### END CODE HERE ###
#%%
# Modify the code below to implement the Adam optimizer
### MODIFY CODE IN THIS BLOCK ###  (≈ 1 line of code to be modified)
import torch.optim as optim
NN = Model(83,32,32,32,1)
criterion = nn.MSELoss()
optimizer = optim.Adam(NN.parameters(), lr=0.01)
### MODIFY CODE IN THIS BLOCK ###
for epoch in range(1000): 
    epoch_loss = 0.0
    for X, Y in dataloader_train:
        Z = NN(X)
        optimizer.zero_grad()
        loss = criterion(Z,Y)
        loss.backward()
        optimizer.step()
        epoch_loss += Z.shape[0] * loss.item()
    print(f'Epoch {epoch+1:4d}: training loss = {epoch_loss/len(data_train):.4f}')

#%%
# Why would it be smart to use ReLU 