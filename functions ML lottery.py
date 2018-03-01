import math
import time
import numpy as np
import h5py
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops


def ldata(m):

    Location = r'C:\Users\01525817\Documents\Marcelo\mega\Deep NN\jogo.xlsx'
    jogo = pd.read_excel(Location)

    # print results
    #print("Accuracy: "  + str(jogo))

    dataset = {}

    X = np.zeros((4,m))
    W = np.zeros((10,m))
    Y = np.zeros((60,m))
    
    W[0][:] = jogo.values[0][0:m]
    W[1][:] = jogo.values[3][:m]
    W[2][:] = jogo.values[4][0:m]
    W[3][:] = jogo.values[5][0:m]
    W[4][:] = jogo.values[7][0:m]
    W[5][:] = jogo.values[8][0:m]
    W[6][:] = jogo.values[9][0:m]
    W[7][:] = jogo.values[10][0:m]
    W[8][:] = jogo.values[11][0:m]
    W[9][:] = jogo.values[12][0:m]

    X[0][:] = jogo.values[0][0:m]
    X[1][:] = jogo.values[3][:m]
    X[2][:] = jogo.values[4][0:m]
    X[3][:] = jogo.values[5][0:m]


    #X = X.astype(int)
    W = W.astype(int)
    
    for i in range(0,m):
        for k in range(4,10):
            Y[W[k][i]-1][i] = 1
            

    dataset["X"] = X
    dataset["Y"] = Y.astype(int)
    
    return dataset

# utils ==================================================

def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def random_mini_batches(X, Y, mini_batch_size = 64):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []

    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches




#flexibilizar
def forward_propagation_for_predict(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    L = len(parameters) // 2
    A = X

    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3'] 
                                                           # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3,Z2) + b3
    
    return Z3

def sigmoid(z):
    """
    Computes the sigmoid of z
    
    Arguments:
    z -- input value, scalar or vector
    
    Returns: 
    results -- the sigmoid of z
    """
    
    ### START CODE HERE ### ( approx. 4 lines of code)
    # Create a placeholder for x. Name it 'x'.
    x = tf.placeholder(tf.float32, name = "x")

    # compute sigmoid(x)
    sigmoid = tf.sigmoid(x)

    # Create a session, and run it. Please use the method 2 explained above. 
    # You should use a feed_dict to pass z's value to x. 
    with tf.Session() as sess:
        # Run session and call the output "result"
        result = sess.run(sigmoid, feed_dict = {x: z})
    
    ### END CODE HERE ###
    
    return result

def cost_sigmoid(logits, labels):
    """
    Computes the cost using the sigmoid cross entropy
    
    Arguments:
    logits -- vector containing z, output of the last linear unit (before the final sigmoid activation)
    labels -- vector of labels y (1 or 0) 
    
    Note: What we've been calling "z" and "y" in this class are respectively called "logits" and "labels" 
    in the TensorFlow documentation. So logits will feed into z, and labels into y. 
    
    Returns:
    cost -- runs the session of the cost (formula (2))
    """
    
    ### START CODE HERE ### 
    
    # Create the placeholders for "logits" (z) and "labels" (y) (approx. 2 lines)
    z = tf.placeholder(tf.float32, name = "z")
    y = tf.placeholder(tf.float32, name = "y")
    
    # Use the loss function (approx. 1 line)
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits = z, labels = y)
    
    # Create a session (approx. 1 line). See method 1 above.
    sess = tf.Session()
    
    # Run the session (approx. 1 line).
    cost = sess.run(cost, feed_dict = {z: logits, y: labels})
    
    # Close the session (approx. 1 line). See method 1 above.
    sess.close()
    
    ### END CODE HERE ###
    
    return cost

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def one_hot_matrix(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j) 
                     will be 1. 
                     
    Arguments:
    labels -- vector containing the labels 
    C -- number of classes, the depth of the one hot dimension
    
    Returns: 
    one_hot -- one hot matrix
    """
    
    ### START CODE HERE ###
    
    # Create a tf.constant equal to C (depth), name it 'C'. (approx. 1 line)
    C = tf.constant(C, name = 'C')
    
    # Use tf.one_hot, be careful with the axis (approx. 1 line)
    one_hot_matrix = tf.one_hot(labels, C, axis=0)
    
    # Create the session (approx. 1 line)
    sess = tf.Session()
    
    # Run the session (approx. 1 line)
    one_hot = sess.run(one_hot_matrix)
    
    # Close the session (approx. 1 line). See method 1 above.
    sess.close()
    
    ### END CODE HERE ###
    
    return one_hot

def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    
    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """

    ### START CODE HERE ### (approx. 2 lines)
    X = tf.placeholder(tf.float32, [n_x, None])
    Y = tf.placeholder(tf.float32, [n_y, None])
    ### END CODE HERE ###
    
    return X, Y

# flexibilizado
def initialize_parameters(layers_dims):
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    
    parameters = {}
    L = len(layers_dims)
    for l in range(1,L):
         parameters['W' + str(l)] = tf.get_variable("W"+str(l), [layers_dims[l],layers_dims[l-1]], initializer = tf.contrib.layers.xavier_initializer())
         parameters['b' + str(l)] = tf.get_variable("b"+str(l), [layers_dims[l], 1], initializer = tf.zeros_initializer())
            
    return parameters

# flexibilizado
def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    A = X
    L = len(parameters) // 2                  # number of layers in the neural network

    for l in range(1,L+1):
        A_prev = A
        Z = tf.add(tf.matmul(parameters['W' + str(l)],A_prev),parameters['b' + str(l)])                         # Z(l) = np.dot(W(l), A(l-1)) + b(l)
        A = tf.nn.relu(Z)                                                                                       # A(l) = relu(Z(l))
            
    return Z

def compute_cost_softmax(Z, Y):
    """
    Computes the cost
    
    Arguments:
    Z -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z)
    labels = tf.transpose(Y)   
    
    ### START CODE HERE ### (1 line of code)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
   
    ### END CODE HERE ###
    
    return cost

#flexibilizado
def model(X_train, Y_train, X_test, Y_test, layers_dims, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    print("learning rate is = " + str(learning_rate))
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of shape (n_x, n_y)
    ### START CODE HERE ### (1 line)
    X, Y = create_placeholders(n_x, n_y)
    ### END CODE HERE ###

    # Initialize parameters
    ### START CODE HERE ### (1 line)
    parameters = initialize_parameters(layers_dims)
    ### END CODE HERE ###
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    ### START CODE HERE ### (1 line)
    Z = forward_propagation(X, parameters)
    ### END CODE HERE ###
    
    # Cost function: Add cost function to tensorflow graph
    ### START CODE HERE ### (1 line)
    cost = compute_cost_softmax(Z, Y)
    ### END CODE HERE ###
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    ### START CODE HERE ### (1 line)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    ### END CODE HERE ###
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            
            
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)
           
            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict = {X: minibatch_X, Y: minibatch_Y})
                ### END CODE HERE ###
                
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions just for 1 class in one column
        correct_prediction = tf.equal(tf.argmax(Z), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters

def save_param(nome, param):
    np.save(nome + '.npy', param)
    return None

def load_param(nome):
    param = np.load(nome + '.npy').item()
    return param

# flexibilizado
def fp_predict(X, parameters):
    
    L = len(parameters) // 2                  # number of layers in the neural network

    params= {}

    for l in range(1,L+1):
        params["W" + str(l)] = tf.convert_to_tensor(parameters["W"+str(l)])
        params["b" + str(l)] = tf.convert_to_tensor(parameters["b"+str(l)])


    n_x = X.shape[0]
    print("size x1 = " + str(n_x))
    x = tf.placeholder(tf.float32, [n_x, None], name = "x1")
    
    z = forward_propagation(x, params)
    
    sess = tf.Session()
    prediction = sess.run(z, feed_dict = {x: X})
        
    return prediction

def predict_softmax(m1,m2,arquivo):

    dados = ldata(m1)
    X = dados["X"]
    Y = dados["Y"]
    x1 = X[:,(m1-m2):]
    y1 = Y[:,(m1-m2):]
    parameters = load_param(arquivo)
    
    Z = fp_predict(x1, parameters)

    print("shape Z = " + str(Z.shape))

    t = np.exp(Z)
    tsum = np.sum(t)

    y_hat = t/tsum

    return x1, y1, y_hat



# utils ==================================================

# rodar DL com tensorflow - jogo mega
def rodar4(L,m,teste,r,ep):

    # ultimo com 10% >>> L = [4, 300, 200, 100, 100, 100, 60] /// P = rodar4(L,2000,10,0.00000005,4000)

    dados = ldata(m)
    X = dados["X"]
    Y = dados["Y"]

    X_train = X[:,0:(m-teste)]
    Y_train = Y[:,0:(m-teste)]*(1/6)

    X_test = X[:,(m-teste):]
    Y_test = Y[:,(m-teste):]*(1/6)
    
    print("size X = " + str(X_train.shape))
    print("X[0][0] = " + str(X_train[0,0]))
    print("X[1][0] = " + str(X_train[1,0]))
    print("X[2][0] = " + str(X_train[2,0]))
    print("X[3][0] = " + str(X_train[3,0]))
    print("size Y = " + str(Y_train.shape))
    print("Y[0] = " + str(Y_train[:,0].T))
    

    parameters = model(X_train, Y_train, X_test, Y_test, layers_dims = L, learning_rate = r,num_epochs = ep)

    return parameters

def rodar(m, r, i, layers_dims):

    params = ldata(m)
    X = params["X"]
    Y = params["Y"]

    #layers_dims = [10, 20, 7, 5, 60] #  5-layer model

    #layers_dims = [10, 1000, 800, 600, 400, 60]


    P = L_layer_model(X, Y, layers_dims, learning_rate = r, num_iterations = i, print_cost=True)

    return P

def rodar2(m, r, i, layers_dims):

    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
    #layers_dims = [12288, 20, 7, 5, 1] #  5-layer model

    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.

    P = L_layer_model(train_x, train_y, layers_dims, learning_rate = r, num_iterations = i, print_cost=True)

    return P

def pausar():
    input("PRESS ENTER TO CONTINUE.")

# rodar DL com tensorflow - figura sinais mao
def rodar3(L):
    
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

    # Flatten the training and test images
    X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
    X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
    # Normalize image vectors
    X_train = X_train_flatten/255.
    X_test = X_test_flatten/255.
    # Convert training and test labels to one hot matrices (testar com função TENSOR)
    Y_train = convert_to_one_hot(Y_train_orig, 6)
    Y_test = convert_to_one_hot(Y_test_orig, 6)

    parameters = model(X_train, Y_train, X_test, Y_test, layers_dims = L)

    return parameters







    

    

    


    


