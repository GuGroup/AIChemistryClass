import numpy as np
import pandas as pd
import torch
import torch.nn as nn
df = pd.read_csv('HW12.csv')
data = df.to_numpy()
# Convert the numpy array to torch tensor
### START CODE HERE ### (≈ 1 line of code)
data = torch.tensor(data)
### END CODE HERE ###
data = data.type(torch.float32)
Y = data[:,:1]
X = data[:,1:]

def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (number of examples, input size)
    Y -- labels of shape (number of examples, output size)
    
    Returns:
    n_0 -- the size of the input layer
    n_1 -- the size of the first hidden layer
    n_2 -- the size of the second hidden layer
    n_3 -- the size of the output layer
    """
    # define the number of units for each layer
    ### START CODE HERE ### (≈ 3 lines of code)
    n_0 = X.shape[1]
    n_1 = 64
    n_2 = 32
    n_3 = 1
    ### END CODE HERE ###
    return n_0, n_1, n_2, n_3

n_0,n_1,n_2,n_3 = layer_sizes(X, Y)

class Model(nn.Module):
    def __init__(self,n_0,n_1,n_2,n_3):
        super().__init__()
        # Make 3 linear layers
        ### START CODE HERE ### (≈ 4 line of code)
        self.L1 = nn.Linear(n_0,n_1)
        self.L2 = nn.Linear(n_1,n_2)
        self.L3 = nn.Linear(n_2,n_3)
        self.act1 = nn.ReLU()
        ### END CODE HERE ###
        
    def forward(self, X):
        # perform forward. With two tanh and sigmoid
        ### START CODE HERE ### (≈ 1 line of code)
        Z1 = self.L1(X)
        A1 = self.act1(Z1)
        Z2 = self.L2(A1)
        A2 = self.act1(Z2)
        Z3 = self.L3(A2)
        ### END CODE HERE ###
        return Z3

NN = Model(n_0,n_1,n_2,n_3)
import torch.optim as optim
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(NN.parameters(), lr=0.1)
for i in range(50000):
    Z = NN(X)
    optimizer.zero_grad()
    loss = criterion(Z,Y)
    loss.backward()
    optimizer.step()
    if i % 2000 == 0:
        print(f'{i} loss: {loss:.3f}')

Z = NN(X)
A = torch.sigmoid(Z)
y_hat = 1*(A>0.5)
accuracy = torch.mean((Y==y_hat).type(torch.float))
print(f'The model accuracy is {accuracy:.3f} %')