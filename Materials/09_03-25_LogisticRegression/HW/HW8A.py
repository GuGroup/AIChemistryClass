# a 
import pandas as pd
df = pd.read_csv('MetalbyComp.csv')
data = df.to_numpy()

y = data[:,0]
x = data[:,1:]

#b
import numpy as np
def logistic(w,b,x):
    return 1/(1+np.exp(-(np.matmul(x,w)+b)))

print(logistic(np.ones(79),1,x).sum())

#c

def cost_function(w,b,x,y):
    f = logistic(w,b,x)
    return -(y*np.log(f)+(1-y)*np.log(1-f)).mean()


print(cost_function(np.ones(79),1,x,y).sum())

# d


def gradient_descent(w,b,x,y,alpha):
    R = logistic(w,b,x)-y
    w -= alpha/x.shape[0]*np.matmul(R,x)
    b -= alpha*R.mean()
    return w,b



# w,b = gradient_descent(np.ones(79),1,x,y,0.01)
# print(w.sum(),b.sum())
# print(cost_function(np.ones(79),1,x,y))
# print(cost_function(w,b,x,y))

# raise

# '''
# w = [0.5,0.1,0.2,0.3]
# b = -8
# J = []
# for i in range(100000):
#     w,b = gradient_descent(w,b,x,y,0.001)
#     J.append(cost_function(w,b,x,y))
# plt.plot(np.arange(len(J)),J)
# plt.xlabel('iteration #')
# plt.ylabel('J')
# '''
import matplotlib.pyplot as plt
w = np.ones(79)
b = 1
J = []
for i in range(100000):
    w,b = gradient_descent(w,b,x,y,1)
    J.append(cost_function(w,b,x,y))
    if len(J) > 2 and J[-2] - J[-1] < 1E-11:
        break
plt.plot(np.arange(len(J)),J)
plt.xlabel('iteration #')
plt.ylabel('J')
plt.show()

# f 

print((y[y_hat>=0.5].sum()+(y[y_hat<0.5]==0).sum())/1130)