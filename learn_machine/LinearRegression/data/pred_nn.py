
# Package imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def clc_accuracy(y_true, y_predict):
    """ use sklearn to calcuate the R2 score"""
    from  sklearn.metrics import r2_score
    score = r2_score(y_true, y_predict)
    return score

def sigmoid(Z):
    """
    Compute the sigmoid of x
    Arguments:
    x -- A scalar or numpy array of any size.
    Return:
    s -- sigmoid(x)
    """
    A = 1.0/(1.0+np.exp(-Z))
    return A

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache
    s = 1.0 / (1.0 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    assert (dZ.shape == Z.shape)
    return dZ

def relu(Z):
    """
    Implement the RELU function.
    Arguments:
    Z -- Output of the linear layer, of any shape
    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    A = np.maximum(0, Z)
    assert (A.shape == Z.shape)
    return A

def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ

def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)

    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    n_x = X.shape[0]  # size of input layer, number of features ?
    n_h = 4
    n_y = Y.shape[0]  # size of output layer
    return (n_x, n_h, n_y)


def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    Returns:
    params -- python dictionary containing your parameters:
    W1 -- weight matrix of shape (n_h, n_x)
    b1 -- bias vector of shape (n_h, 1)
    W2 -- weight matrix of shape (n_y, n_h)
    b2 -- bias vector of shape (n_y, 1)
    """
    np.random.seed(2)  # we set up a seed so that your output matches ours although the initialization is random.

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters


def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]  #
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Implement Forward Propagation to calculate A2 (probabilities)
    Z1 = np.dot(W1, X) + b1
    #A1 = np.tanh(Z1)
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    #A2 = sigmoid(Z2)
    A2 = Z2

    assert (A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    return A2, cache


def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (13)
    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2
    Returns:
    cost -- cross-entropy cost given equation (13)
    """
    m = Y.shape[1]  # number of example

    # Compute the cross-entropy cost
    #logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
    logprobs = - 0.5 * np.multiply(A2-Y, A2-Y)

    cost = - 1.0 / m * np.sum(logprobs)
    cost = np.squeeze(cost)  # makes sure cost is the dimension we expect.

    assert (isinstance(cost, float))
    return cost


def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.
    Arguments:
    parameters -- python dictionary containing our parameters(W1,b1,W2,b2)
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]

    # First, retrieve W1 and W2 from the dictionary "parameters".
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    # Retrieve also A1 and A2 from dictionary "cache".
    A1 = cache["A1"]
    A2 = cache["A2"]
    Z1 = cache["Z1"]

    # Backward propagation: calculate dW1, db1, dW2, db2.
    dZ2 = A2 - Y
    dW2 = 1.0/m * np.dot(dZ2, A1.T)
    db2 = 1.0/m * np.sum(dZ2, axis=1, keepdims=True)   #add each row, keep as 2D array
    dA1 = np.dot(W2.T, dZ2)

    #dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dZ1 = sigmoid_backward(dA1, Z1)

    dW1 = 1.0/m * np.dot(dZ1, X.T)
    db1 = 1.0/m * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    return grads


def update_parameters(parameters, grads, learning_rate=1.2):
    """
    Updates parameters using the gradient descent update rule given above
    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients
    Returns:
    parameters -- python dictionary containing your updated parameters
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    # Update rule for each parameter
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]  # (n_x, n_h, n_y) = layer_sizes(X, Y)
    n_y = layer_sizes(X, Y)[2]

    # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)

        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y, parameters)

        #print ("A2 = " + str(A2))

        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)

        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads, learning_rate=1.2)

        # Print the cost every 1000 iterations
        if print_cost and i % 100 == 0:
        #if i < 60:
            print ("Cost after iteration %i: %f" % (i, cost))
    return parameters


def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    Arguments:
    parameters -- python dictionary containing your parameters
    X -- input data of size (n_x, m)
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """

    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A2, cache = forward_propagation(X, parameters)  # cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"

    return A2

def load_dataset():
    # parting dataset
    data = pd.read_csv('housing.csv')
    prices = data['MEDV'].values
    prices = prices.reshape(-1, 1)
    # prices = prices / prices.max()
    prices = prices / 100000
    features = data.drop('MEDV', axis=1).values
    f_max = 40.0
    features = features / f_max

    print "Boston housing dataset has {} data points with {} variables each.".format(*data.shape)
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(features, prices, test_size=0.2, random_state=10)

    X_train = X_train.T
    Y_train = Y_train.T
    X_test = X_test.T
    Y_test = Y_test.T
    return X_train, X_test, Y_train, Y_test


X_train, X_test, Y_train, Y_test = load_dataset()
shape_X = X_train.shape
shape_Y = Y_train.shape
m = X_train.shape[1]  # training set size

print ('The shape of X train is: ' + str(shape_X))
print ('The shape of Y train is: ' + str(shape_Y))
print ('I have m = %d training examples!' % (m))


# Build a model with a n_h-dimensional hidden layer
parameters = nn_model(X_train, Y_train, n_h = 4, num_iterations = 20000, print_cost=True)

pred_test = predict(parameters, X_test)
pred_err = Y_test - pred_test
Y_test = Y_test.tolist()
pred_test = pred_test.tolist()
Y_test = map(list, zip(*Y_test))
pred_test = map(list, zip(*pred_test))
r2_score = clc_accuracy(Y_test, pred_test)

print "nn_model has R^2 score {:,.2f} on test data".format(r2_score)
#print pred_test

client_data = np.array([(5, 17, 15), (4, 32, 22), (8, 3, 12)])
client_data = client_data.T
predicted_price = predict(parameters, client_data)
for i, price in enumerate(predicted_price):
    print "Predicted selling price for Client {}'s home: $" + str(predicted_price[i]*100000)

xn = range(0, len(Y_test))
#plt.scatter(xn, pred_err)
plt.figure(1)
plt.scatter(xn, Y_test)
plt.scatter(xn, pred_test, c='black')
plt.scatter(xn, pred_err, c='red')
plt.show()







