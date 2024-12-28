# https://youtu.be/Q3vv7Wc1g4g

import numpy as np
data = np.array([[-0.87,2,0,0,0],[-1.08,2,1,0,0],[-1.30,2,2,0,0],[-1.52,2,3,0,0],[-1.73,2,4,0,0],[-1.95,2,5,0,0],[-2.16,2,6,0,0],[-2.37,2,7,0,0],[-2.59,2,8,0,0],[-2.80,2,9,0,0],[-3.01,2,10,0,0],[-3.23,2,11,0,0],[-3.44,2,12,0,0],[-3.68,2,13,0,0],[-3.89,2,14,0,0],[-4.08,2,15,0,0],[-4.30,2,16,0,0],[-4.51,2,17,0,0],[-4.72,2,18,0,0],[-7.22,2,30,0,0],[-1.41,3,0,1,0],[-1.59,3,1,1,0],[-1.81,3,2,1,0],[-1.78,3,2,1,0],[-2.02,3,3,1,0],[-1.99,3,3,1,0],[-1.97,3,3,1,0],[-2.23,3,4,1,0],[-2.20,3,4,1,0],[-2.20,3,4,1,0],[-2.19,3,4,1,0],[-2.70,3,6,1,0],[-2.68,3,6,1,0],[-6.15,3,22,1,0],[-6.09,3,22,1,0],[-7.25,3,27,1,0],[-1.74,4,0,0,1],[-1.92,4,1,0,1],[-1.84,4,0,2,0],[-2.14,4,2,0,1],[-2.06,4,1,2,0],[-2.09,4,1,2,0],[-2.09,4,2,0,1],[-2.33,4,3,0,1],[-2.22,4,2,2,0],[-2.27,4,2,2,0],[-2.31,4,2,2,0],[-2.28,4,3,0,1],[-2.21,4,2,2,0],[-2.19,4,2,2,0],[-2.23,4,3,0,1],[-2.55,4,4,0,1],[-2.12,5,0,1,1],[-2.28,5,1,1,1],[-2.32,5,1,1,1],[-2.24,5,1,1,1],[-2.25,5,0,3,0],[-2.62,5,2,1,1],[-2.51,5,1,3,0],[-2.36,5,1,3,0],[-2.34,6,0,0,2],[-2.50,6,1,0,2],[-2.95,6,2,0,2],[-2.75,6,6,0,2],[-2.45,9,0,1,3],[-2.57,10,0,0,4],[-2.60,12,0,2,4]])

#a
x = data[:,1:]
y = data[:,0]

# b
def linear(w,b,x):
    y_hat = []
    for xx in x:
        y_hat.append(np.dot(w,xx)+b)
    y_hat = np.array(y_hat)
    return y_hat

yhat = linear([0.5,0.1,0.2,0.3],-8,x)

# c
def linear(w,b,x):
    return np.matmul(x,w)+b

yhat = linear([0.5,0.1,0.2,0.3],-8,x)
print(yhat)

# c
import matplotlib.pyplot as plt
def draw(w,b,x,y):
    plt.scatter(x,y)
    plt.xlim(0,35)
    plt.ylim(-8,0)
    x_line = np.array([0,35])
    yhat = linear(w,b,x_line)
    plt.plot(x_line,yhat)
    
#draw(0.5,-8,x,y)

# d

def cost_function(w,b,x,y):
    J = ((linear(w,b,x)-y)**2).mean()/2
    # your code here
    return J


def gradient_descent(w,b,x,y,alpha):
    R = linear(w,b,x)-y
    w -= alpha/x.shape[0]*np.matmul(R,x)
    b -= alpha*R.mean()
    return w,b



w,b = gradient_descent([0.5,0.1,0.2,0.3],-8,x,y,0.01)
print(w,b)
print(cost_function([0.5,0.1,0.2,0.3],-8,x,y))
print(cost_function(w,b,x,y))

'''
w = [0.5,0.1,0.2,0.3]
b = -8
J = []
for i in range(100000):
    w,b = gradient_descent(w,b,x,y,0.001)
    J.append(cost_function(w,b,x,y))
plt.plot(np.arange(len(J)),J)
plt.xlabel('iteration #')
plt.ylabel('J')
'''
# e

w = [0.5,0.1,0.2,0.3]
b = -8
J = []
for i in range(100000):
    w,b = gradient_descent(w,b,x,y,0.001)
    J.append(cost_function(w,b,x,y))
    if len(J) > 2 and J[-2] - J[-1] < 1E-11:
        break
plt.plot(np.arange(len(J)),J)
plt.xlabel('iteration #')
plt.ylabel('J')
plt.show()
# f

x = data[:,1:]
y = data[:,0]

x = (x-x.mean(0))/x.std(0)

w = [0.5,0.1,0.2,0.3]
b = -8
J = []
for i in range(100000):
    w,b = gradient_descent(w,b,x,y,0.001)
    J.append(cost_function(w,b,x,y))
    if len(J) > 2 and J[-2] - J[-1] < 1E-11:
        break
plt.plot(np.arange(len(J)),J)
plt.xlabel('iteration #')
plt.ylabel('J')
plt.show()