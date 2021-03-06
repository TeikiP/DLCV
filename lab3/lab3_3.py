from __future__ import print_function

import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop



#print('tensorflow:', tf.__version__)
#print('keras:', keras.__version__)


#load (first download if necessary) the MNIST dataset
# (the dataset is stored in your home direcoty in ~/.keras/datasets/mnist.npz
#  and will take  ~11MB)
# data is already split in train and test datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train : 60000 images of size 28x28, i.e., x_train.shape = (60000, 28, 28)
# y_train : 60000 labels (from 0 to 9)
# x_test  : 10000 images of size 28x28, i.e., x_test.shape = (10000, 28, 28)
# x_test  : 10000 labels
# all datasets are of type uint8

#To input our values in our network Dense layer, we need to flatten the datasets, i.e.,
# pass from (60000, 28, 28) to (60000, 784)
#flatten images
num_pixels = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], num_pixels)
x_test = x_test.reshape(x_test.shape[0], num_pixels)

#Convert to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Normalize inputs from [0; 255] to [0; 1]
x_train = x_train / 255
x_test = x_test / 255


#Convert class vectors to binary class matrices ("one hot encoding")
## Doc : https://keras.io/utils/#to_categorical
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)


num_classes = y_train.shape[1]
epochs = 100
neurones_output = 10
neurones_hidden = 64
batch_size = 64



#Let start our work: creating a neural network


#Let start our work: creating a neural network
#First, we just use a single neuron. 

# creation du modele
model = Sequential()

model.add(Dense(neurones_hidden, activation='relu', input_dim=num_pixels))
model.add(Dense(neurones_output, activation='softmax'))

# compiler le modele
#model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy']) # binary, stochastic gradient descent
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # multiclass

# entrainer le reseau
hist = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))

# evaluer le modele
score = model.evaluate(x_test, y_test)
print('Score apres' , epochs, 'epochs =', score)

# prediction a partir du modele
#y_inference = model.predict(x_test)
#print(y_inference.shape)


#import matplotlib.pyplot as plt

#plt.plot(hist.history['acc'])
#plt.plot(hist.history['val_acc'])
#plt.title('model accuracy')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
t')
plt.show()

plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
