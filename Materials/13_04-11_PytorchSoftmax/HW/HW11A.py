# https://github.com/Gurupradeep/deeplearning.ai-Assignments/blob/master/Neural-networks-Deep-learning/Week3/Planar%2Bdata%2Bclassification%2Bwith%2Bone%2Bhidden%2Blayer%2Bv4.ipynb
#%%
import numpy as np
import pandas as pd
df = pd.read_csv('HW11.csv')
data = df.to_numpy()
Y = data[:,0]
X = data[:,1:]
#%%
### START CODE HERE ### (≈ 3 lines of code)
shape_X = X.shape
shape_Y = Y.shape
m = X.shape[0]
### END CODE HERE ###

print ('The shape of X is: ' + str(shape_X))
print ('The shape of Y is: ' + str(shape_Y))
print ('I have m = %d training examples!' % (m))
assert shape_X == (1130, 79), 'Wrong answer!'
assert shape_Y == (1130, ), 'Wrong answer!'
assert m == 1130, 'Wrong answer!'

#%%
def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (number of examples, input size)
    Y -- labels of shape (number of examples, output size)
    
    Returns:
    n_0 -- the size of the input layer
    n_1 -- the size of the hidden layer
    n_2 -- the size of the output layer
    """
    ### START CODE HERE ### (≈ 3 lines of code)
    n_0 = X.shape[1]
    n_1 = 4
    n_2 = 1
    ### END CODE HERE ###
    return (n_0, n_1, n_2)

(n_0, n_1, n_2) = layer_sizes(X, Y)
print("The size of the input layer is: n_0 = " + str(n_0))
print("The size of the hidden layer is: n_1 = " + str(n_1))
print("The size of the output layer is: n_2 = " + str(n_2))
assert n_0 == 79, 'Wrong answer!'
assert n_1 == 4, 'Wrong answer!'
assert n_2 == 1, 'Wrong answer!'

#%%
def initialize_parameters(n_0, n_1, n_2):
    """
    Argument:
    n_0 -- size of the input layer
    n_1 -- size of the hidden layer
    n_2 -- size of the output layer
    
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_0, n_1)
                    b1 -- bias vector of shape (n_1)
                    W2 -- weight matrix of shape (n_1, n_2)
                    b2 -- bias vector of shape (n_2)
    """
    
    np.random.seed(2) # we set up a seed so that your output matches ours although the initialization is random.
    
    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = np.random.randn(n_0,n_1)*0.01
    b1 = np.zeros((n_1,))
    W2 = np.random.randn(n_1,n_2)*0.01
    b2 = np.zeros((n_2,))
    ### END CODE HERE ###
    

    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    

    return parameters

parameters = initialize_parameters(n_0, n_1, n_2)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
assert (parameters["W1"].shape == (n_0, n_1)), 'Wrong answer!'
assert (parameters["b1"].shape == (n_1, )), 'Wrong answer!'
assert (parameters["W2"].shape == (n_1, n_2)), 'Wrong answer!'
assert (parameters["b2"].shape == (n_2, )), 'Wrong answer!'

#%%

def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (m, n_0)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    ### END CODE HERE ###
    
    # Implement Forward Propagation to calculate A2 (probabilities)
    ### START CODE HERE ### (≈ 4 lines of code)
    Z1 = X@W1 + b1
    A1 = np.tanh(Z1)
    Z2 = A1@W2 + b2
    A2 = 1/(1+np.exp(-Z2))
    ### END CODE HERE ###
    
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache


A2, cache = forward_propagation(X, parameters)
print(A2.shape)
assert A2.shape == (1130,1), 'Wrong answer!'
assert np.round(np.mean(A2),1) == 0.5, 'Wrong answer!'

#%%

def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (13)
    
    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (number of examples, 1)
    Y -- "true" labels vector of shape (number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2
    
    Returns:
    cost -- cross-entropy cost given equation (13)
    """
    # Remove the feature (the last) dimension of the A2
    ### START CODE HERE ### (≈ 1 line of code)
    A2 = A2.reshape(-1)
    ### END CODE HERE ###
    
    # Compute the cross-entropy cost
    ### START CODE HERE ### (≈ 2 lines of code)
    L = -Y*np.log(A2) - (1-Y)*Y*np.log(1-A2)
    J = np.mean(L)
    ### END CODE HERE ###
    
    return J

J = compute_cost(A2, Y, parameters)
print("cost = " + str(J))

assert isinstance(J, float), 'Wrong answer!'
assert np.round(J,2) == 0.37

#%%

def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.
    
    Arguments:
    parameters -- python dictionary containing our parameters 
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (number of examples, n_0)
    Y -- "true" labels vector of shape (number of examples)
    
    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = Y.shape[0]
    # Retrieve also A1 and A2 from dictionary "cache".
    ### START CODE HERE ### (≈ 2 lines of code)
    A1 = cache["A1"]
    A2 = cache["A2"]
    ### END CODE HERE ###

    # Add a dimension to Y
    ### START CODE HERE ### (≈ 1 lines of code)
    Y = Y.reshape(-1,1)
    ### END CODE HERE ###

    # Backward propagation: calculate dW1, db1, dW2, db2. 
    ### START CODE HERE ### (≈ 6 lines of code, corresponding to 6 equations on slide above)
    dLdZ2 = A2-Y
    dJdW2 = A1.T@dLdZ2/m
    dJdb2 = dLdZ2.mean(0)
    dLdZ1 = dLdZ2*(1-A1**2)
    dJdW1 = X.T@dLdZ1/m
    dJdb1 = dLdZ1.mean(0)
    ### END CODE HERE ###
    
    grads = {"dJdW1": dJdW1,
             "dJdb1": dJdb1,
             "dJdW2": dJdW2,
             "dJdb2": dJdb2}
    
    return grads

grads = backward_propagation(parameters, cache, X, Y)
print ("dJdW1 = "+ str(grads["dJdW1"]))
print ("dJdb1 = "+ str(grads["dJdb1"]))
print ("dJdW2 = "+ str(grads["dJdW2"]))
print ("dJdb2 = "+ str(grads["dJdb2"]))
assert grads["dJdW2"].shape == (4,1), 'Wrong answer!'
assert grads["dJdb2"].shape == (1,), 'Wrong answer!'
assert grads["dJdW1"].shape == (79,4), 'Wrong answer!'
assert grads["dJdb1"].shape == (4,), 'Wrong answer!'
assert np.round(np.mean(grads["dJdW1"]),5) == -0.00039, 'Wrong answer!'

#%%

# GRADED FUNCTION: update_parameters

def update_parameters(parameters, grads, learning_rate = 0.1):
    """
    Updates parameters using the gradient descent update rule given above
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients 
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    # Retrieve each parameter from the dictionary "parameters"
    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    ### END CODE HERE ###
    
    # Retrieve each gradient from the dictionary "grads"
    ### START CODE HERE ### (≈ 4 lines of code)
    dW1 = grads["dJdW1"]
    db1 = grads["dJdb1"]
    dW2 = grads["dJdW2"]
    db2 = grads["dJdb2"]
    ## END CODE HERE ###
    
    # Update rule for each parameter
    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2
    ### END CODE HERE ###
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

parameters = update_parameters(parameters, grads)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

#%%

def nn_model(X, Y, n_1, num_iterations = 100000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (number of examples, n_0)
    Y -- labels of shape (number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 10000 iterations
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    np.random.seed(3)
    n_0, _, n_2 = layer_sizes(X, Y)
    
    # Initialize parameters. Inputs: "n_0, n_1, n_2"
    ### START CODE HERE ### (≈ 1 lines of code)
    parameters = initialize_parameters(n_0,n_1,n_2)
    ### END CODE HERE ###
    
    # Loop (gradient descent)

    for i in range(0, num_iterations):
         
        ### START CODE HERE ### (≈ 4 lines of code)
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X,parameters)
        
        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2,Y,parameters)
 
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters,cache,X,Y)
 
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters,grads)
        ### END CODE HERE ###
        
        # Print the cost every 1000 iterations
        if print_cost and i % 10000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

parameters = nn_model(X, Y, 4, num_iterations=100000, print_cost=True)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

#%%

# GRADED FUNCTION: predict

def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (m, n_0)
    
    Returns
    y_hat -- vector of predictions of our model 
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    ### START CODE HERE ### (≈ 2 lines of code)
    A2, cache = forward_propagation(X,parameters)
    y_hat = 1*(A2>0.5)
    ### END CODE HERE ###
    
    # Remove the last diemsnion of the y_hat
    ### START CODE HERE ### (≈ 1 lines of code)
    y_hat = y_hat.reshape(-1)
    ### END CODE HERE ###    

    return y_hat

y_hat = predict(parameters, X)
print("predictions mean = " + str(np.mean(y_hat)))

assert np.round(np.mean(y_hat),3) == 0.545, 'Wrong answer!'
#%%
def caclulate_accuracy(y_hat, Y):
    """
    Using the y_hat and Y, calculate the accuracy
    
    Arguments:
    y_hat -- predicted Y values (m,)
    Y -- actual Y values (m, )
    
    Returns
    accuracy -- a float value of accuracy
    """
    
    # 
    ### START CODE HERE ### (≈ 2 lines of code)
    accuracy = np.mean(Y==y_hat)
    ### END CODE HERE ###
    
    return accuracy

accuracy = caclulate_accuracy(y_hat, Y)
print("Accuracy = " + str(np.mean(y_hat)))
assert np.round(np.mean(accuracy),3) == 0.837, 'Wrong answer!'

#%%

# This may take about 2 minutes to run
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):
    parameters = nn_model(X, Y, n_h, num_iterations = 100000)
    y_hat = predict(parameters, X)
    accuracy = caclulate_accuracy(y_hat, Y)
    print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
raise
#%%
