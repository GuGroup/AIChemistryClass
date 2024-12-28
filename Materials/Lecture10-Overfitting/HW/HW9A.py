
import numpy as np
data = np.array([[-0.87,2,0,0,0],[-1.08,2,1,0,0],[-1.30,2,2,0,0],[-1.52,2,3,0,0],[-1.73,2,4,0,0],[-1.95,2,5,0,0],[-2.16,2,6,0,0],[-2.37,2,7,0,0],[-2.59,2,8,0,0],[-2.80,2,9,0,0],[-3.01,2,10,0,0],[-3.23,2,11,0,0],[-3.44,2,12,0,0],[-3.68,2,13,0,0],[-3.89,2,14,0,0],[-4.08,2,15,0,0],[-4.30,2,16,0,0],[-4.51,2,17,0,0],[-4.72,2,18,0,0],[-7.22,2,30,0,0],[-1.41,3,0,1,0],[-1.59,3,1,1,0],[-1.81,3,2,1,0],[-1.78,3,2,1,0],[-2.02,3,3,1,0],[-1.99,3,3,1,0],[-1.97,3,3,1,0],[-2.23,3,4,1,0],[-2.20,3,4,1,0],[-2.20,3,4,1,0],[-2.19,3,4,1,0],[-2.70,3,6,1,0],[-2.68,3,6,1,0],[-6.15,3,22,1,0],[-6.09,3,22,1,0],[-7.25,3,27,1,0],[-1.74,4,0,0,1],[-1.92,4,1,0,1],[-1.84,4,0,2,0],[-2.14,4,2,0,1],[-2.06,4,1,2,0],[-2.09,4,1,2,0],[-2.09,4,2,0,1],[-2.33,4,3,0,1],[-2.22,4,2,2,0],[-2.27,4,2,2,0],[-2.31,4,2,2,0],[-2.28,4,3,0,1],[-2.21,4,2,2,0],[-2.19,4,2,2,0],[-2.23,4,3,0,1],[-2.55,4,4,0,1],[-2.12,5,0,1,1],[-2.28,5,1,1,1],[-2.32,5,1,1,1],[-2.24,5,1,1,1],[-2.25,5,0,3,0],[-2.62,5,2,1,1],[-2.51,5,1,3,0],[-2.36,5,1,3,0],[-2.34,6,0,0,2],[-2.50,6,1,0,2],[-2.95,6,2,0,2],[-2.75,6,6,0,2],[-2.45,9,0,1,3],[-2.57,10,0,0,4],[-2.60,12,0,2,4]])
def linear(w,b,x):
    return np.matmul(x,w)+b

# a 
np.random.shuffle(data)
# b
ntrain = int(data.shape[0]*0.8)
nval = int(data.shape[0]*0.8)

data_train = data[:ntrain,:]
data_val = data[ntrain:ntrain+nval,:]
data_test = data[ntrain+nval:,:]
# c
y_train = data_train[:,0]
x_train = data_train[:,1:]
y_val = data_train[:,0]
x_val = data_train[:,1:]
y_test = data_train[:,0]
x_test = data_train[:,1:]

x_mean = x_train.mean(0)
x_std = x_train.std(0)

x_train = (x_train-x_mean)/x_std
x_val = (x_val-x_mean)/x_std
x_test = (x_test-x_mean)/x_std
#d


def cost_function(w,b,x,y,lamb):
    J = ((linear(w,b,x)-y)**2).mean()/2
    J += lamb/2/x.shape[0]*(w**2).sum()
    J += lamb/2/x.shape[0]*(b**2)
    return  J


print(cost_function(np.ones(4),1,x_train,y_train,1).sum())

def gradient_descent(w,b,x,y,alpha,lamb):
    R = linear(w,b,x)-y
    w -= alpha/x.shape[0]*np.matmul(R,x) + lamb/x.shape[0]*w
    b -= alpha*R.mean() + lamb/x.shape[0]*b
    return w,b

w,b = gradient_descent(np.ones(4),1,x_train,y_train,0.01,1)
print(w.sum(),b.sum())
# e


alpha = 0.01
J_vals = []
lambdas = 10**(np.arange(-25,0,0.5))
ws = []
bs = []
for lamb in lambdas:
    w = np.ones(4)
    b = 1
    J_train_old = float('inf')
    for i in range(100000):
        w,b = gradient_descent(w,b,x_train,y_train,alpha,lamb)
        J_train = cost_function(w,b,x_train,y_train,lamb)
        if abs(J_train_old - J_train) < 1E-11:
            break
        J_train_old = J_train
    ws.append(w)
    bs.append(b)
    J_val = cost_function(w,b,x_val,y_val,lamb)
    J_vals.append(J_val)

min_lambda = lambdas[np.argmin(J_vals)]
w = ws[np.argmin(J_vals)]
b = bs[np.argmin(J_vals)]
J_test = cost_function(w,b,x_test,y_test,min_lambda)