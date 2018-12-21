import numpy as np

#In this first part, we just prepare our data (mnist) 
#for training and testing

import keras
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).T
X_test = X_test.reshape(X_test.shape[0], num_pixels).T
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
X_train  = X_train / 255
X_test  = X_test / 255



# one-hot encode labels
digits = 10

def one_hot_encode(y, digits):
    examples = y.shape[0]
    y = y.reshape(1, examples)
    Y_new = np.eye(digits)[y.astype('int32')]  #shape (1, 70000, 10)
    Y_new = Y_new.T.reshape(digits, examples)
    return Y_new

y_train=one_hot_encode(y_train, digits)
y_test=one_hot_encode(y_test, digits)
m = X_train.shape[1] #number of examples

nn = 64 # number of neurones in the hidden layer
epochs = 500
lr = 1

#Now, we shuffle the training set
np.random.seed(138)
shuffle_index = np.random.permutation(m)
X_train, y_train = X_train[:,shuffle_index], y_train[:,shuffle_index]

# #Display one image and corresponding label 
# import matplotlib
# import matplotlib.pyplot as plt
# i = 35
# print('y[{}]={}'.format(i, y_train[:,i]))
# plt.imshow(X_train[:,i].reshape(28,28), cmap = matplotlib.cm.binary)
# plt.axis("off")
# plt.show()

def sigmoid(x) :
    return 1. / (1. + np.exp(-x))
    
def computeLoss(y, A) :
    loss = np.sum(np.multiply(y, np.log(A)))

    return - loss / A.shape[1]
    
# initial values
w, _ = X_train.shape
Wh = np.random.randn(nn, w) * .01
Wo = np.random.randn(10, nn) * .01
Bh = np.zeros((1, 1))
Bo = np.zeros((1, 1))

# number of iterations
for i in range(1, epochs+1) :

    # Forward propagation - Hidden Layer
    Zh = np.matmul(Wh, X_train) + Bh # weighted input + bias
    Ah = sigmoid(Zh) # activation function
    

    # Forward propagation - Output Layer
    Zo = np.matmul(Wo, Ah) + Bo # weighted input + bias
    Ao = sigmoid(Zo) # activation function
    
    # Current loss ratio
    if i % 10 == 0 :
        loss = computeLoss(y_train, Ao)
        print(i, loss)
    
    
    # Back propagation - Hidden Layer
    cost = Ao - y_train  
    sig_Zh = sigmoid(Zh)
    sub_dWh = np.matmul(np.transpose(Wo), cost) * sig_Zh * (1 - sig_Zh)
    
    dWh = 1.0 / m * np.matmul(sub_dWh, np.transpose(X_train))
    dBh = 1.0 / m * np.sum(sub_dWh)
    
    Wh -= lr * dWh
    Bh -= lr * dBh
    
    # Back propagation - Output Layer
    dWo = 1.0 / m * np.matmul(cost, np.transpose(Ah))
    dBo = 1.0 / m * np.sum(cost)
    
    Wo -= lr * dWo
    Bo -= lr * dBo

