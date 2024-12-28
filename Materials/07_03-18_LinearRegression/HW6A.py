import numpy as np
data = np.array([[2,-0.87024],[3,-1.08469],[4,-1.30122],[5,-1.52085],[6,-1.73116],[7,-1.94561],[8,-2.16213],[9,-2.36519],[10,-2.58689],[11,-2.80031],[12,-3.01372],[13,-3.22714],[14,-3.44056],[15,-3.67573],[16,-3.88396],[17,-4.0808],[18,-4.29526],[19,-4.50764],[20,-4.72209],[32,-7.21678]])

#a
x = data[:,0]
y = data[:,1]

# b
def linear(w,b,x):
    return x*w+b

yhat = linear(0.5,-8,x)
print(yhat.sum())

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

print(cost_function(0.5,-8,x,y))

# e
def gradient_descent(w,b,x,y,alpha):
    w_tmp = w - alpha*((w*x+b-y)*x).mean()
    b_tmp = b - alpha*(w*x+b-y).mean()
    w_new = w_tmp
    b_new = b_tmp
    return w_new, b_new

w,b = gradient_descent(0.5,-8,x,y,0.01)
print(w,b)
print(cost_function(w,b,x,y))
raise
# f
w,b = gradient_descent(0.5,-8,x,y,0.011)
print(cost_function(w,b,x,y))
w,b = gradient_descent(0.5,-8,x,y,0.001)
print(cost_function(w,b,x,y))

# g
w = 0.5
b = -8
for i in range(20000):
    w,b = gradient_descent(w,b,x,y,0.001)
    if i%1000 == 0:
        draw(w,b,x,y)
draw(w,b,x,y)