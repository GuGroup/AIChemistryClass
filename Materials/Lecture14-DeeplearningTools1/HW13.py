from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

class MaterialsDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        data = df.to_numpy()
        data = torch.tensor(data,dtype=torch.float32)
        self.Y = data[:,:1]
        self.X = data[:,1:]

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        return self.X[idx,:], self.Y[idx,:]
data = MaterialsDataset('HW13.csv')

ntrain = int(len(data)*0.8)
nval_test = int(len(data)*0.1)
data_train, data_val, data_test = random_split(data,[ntrain,nval_test,nval_test])

dataloader_train = DataLoader(data_train, batch_size=64, shuffle=True)
dataloader_val = DataLoader(data_val, batch_size=64, shuffle=True)
dataloader_test = DataLoader(data_test, batch_size=64, shuffle=True)

import torch.nn as nn

class Model(nn.Module):
    def __init__(self,n_0,n_1,n_2,n_3):
        super().__init__()
        # Make 3 linear layers
        ### START CODE HERE ### (≈ 4 line of code)
        self.L1 = nn.Linear(n_0,n_1)
        self.bn1 = nn.BatchNorm1d(n_1)
        self.L2 = nn.Linear(n_1,n_2)
        self.L3 = nn.Linear(n_2,n_3)
        self.act1 = nn.ReLU()
        ### END CODE HERE ###
        
    def forward(self, X):
        # perform forward. With two tanh and sigmoid
        ### START CODE HERE ### (≈ 1 line of code)
        Z1 = self.L1(X)
        Z1 = self.bn1(Z1)
        A1 = self.act1(Z1)
        Z2 = self.L2(A1)
        A2 = self.act1(Z2)
        Z3 = self.L3(A2)
        ### END CODE HERE ###
        return Z3


import torch.optim as optim
NN = Model(79,64,32,1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(NN.parameters(), lr=0.1)

for i in range(1000): 
    for X, Y in dataloader_train:
        Z = NN(X)
        optimizer.zero_grad()
        loss = criterion(Z,Y)
        loss.backward()
        optimizer.step()
    print(i,loss)
