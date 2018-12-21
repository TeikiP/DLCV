from __future__ import print_function

import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import RMSprop

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


print('tensorflow:', tf.__version__)
print('keras:', keras.__version__)


#load (first download if necessary) the CIFAR10 dataset
# data is already split in train and test datasets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#To input our values in our network Conv2D layer, we need to reshape the datasets, i.e.,
# pass from (60000, 28, 28) to (60000, 28, 28, 1) where 1 is the number of channels of our images
img_rows, img_cols = x_train.shape[1], x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)

#Convert to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Normalize inputs from [0; 255] to [0; 1]
x_train = x_train / 255
x_test = x_test / 255


print('x_train.shape=', x_train.shape)
print('x_test.shape=', x_test.shape)
print('y_train.shape=', y_train.shape)
print('y_test.shape=', y_test.shape)

num_classes = 10
epochs = 10
batch_size = 64
filters = 64

#Convert class vectors to binary class matrices ("one hot encoding")
## Doc : https://keras.io/utils/#to_categorical
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
# num_classes is computed automatically here
# but it is dangerous if y_test has not all the classes
# It would be better to pass num_classes=np.max(y_train)+1



#Let start our work: creating a convolutional neural network

# creation du modele
model = Sequential()

# ajout des couches
model.add(Conv2D(filters, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
model.add(Conv2D(filters, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

# compiler le modele
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # multiclass

# entrainer le reseau
hist = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))

# evaluer le modele
score = model.evaluate(x_test, y_test)
print('Score apres' , epochs, 'epochs =', score)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# utilisation du modele
y = model.predict(x_test)

# comparaison des resultats uniquement sur la classe valide
y_err1 = y * y_test

# comparaison des resultats sur les 10 classes
y_err2 = np.absolute(y - y_test);

# sommes des resultats des comparaisons
y_err1 = np.sum(y_err1, axis=1)
y_err2 = np.sum(y_err2, axis=1)

# tris des comparaisons
y_err1_sorted = np.argsort(y_err1)
y_err2_sorted = np.argsort(y_err2)

# Affichage des 10 images les moins bien classees

for i in range(10) :
  plt.imshow(x_test[y_err1_sorted[i],:].reshape(x_test.shape[1], x_test.shape[2], 3), cmap = matplotlib.cm.binary)
  plt.axis("off")
  plt.show()
  
  #plt.imshow(x_test[y_err2_sorted[y_err2.size - i - 1],:].reshape(x_test.shape[1], x_test.shape[2], 3), cmap = matplotlib.cm.binary)
  #plt.axis("off")
  #plt.show()


