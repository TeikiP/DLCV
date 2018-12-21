from __future__ import print_function
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Conv3D, MaxPooling3D, Flatten, Add, Concatenate
from keras.engine.input_layer import Input
print('Keras version : ', keras.__version__)

#############################################
############## Make the model ###############
#############################################


def make_one_branch_model(temporal_dim, width, height, channels, nb_class):
    #Build the 'one branch' model and compile it.
    input_shape = (temporal_dim, width, height, channels)
    
    #Use the following optimizer
    sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)    
    
    # creation du modele
    model = Sequential()

    # ajout des couches
    model.add(Conv3D(30, kernel_size=(3,3,3), padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    
    model.add(Conv3D(60, kernel_size=(3,3,3), padding='same', activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    
    model.add(Conv3D(80, kernel_size=(3,3,3), padding='same', activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    
    model.add(Flatten())
    
    model.add(Dense(500, activation='relu'))    
    model.add(Dense(nb_class, activation='softmax'))

    # compiler le modele
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


def make_model(temporal_dim, width, height, nb_class):
    #Build the siamese model and compile it.
    rgb_input_shape = (temporal_dim, width, height, 3)
    rgb_input = Input(shape=rgb_input_shape)
    
    flow_input_shape = (temporal_dim, width, height, 2)
    flow_input = Input(shape=flow_input_shape)
    
    #Use the following optimizer
    sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)
     
    # creation des couches RGB
    x1 = Conv3D(30, kernel_size=(3,3,3), padding='same', activation='relu')(rgb_input)
    x1 = MaxPooling3D(pool_size=(2, 2, 2))(x1)
    
    x1 = Conv3D(60, kernel_size=(3,3,3), padding='same', activation='relu')(x1)
    x1 = MaxPooling3D(pool_size=(2, 2, 2))(x1)
    
    x1 = Conv3D(80, kernel_size=(3,3,3), padding='same', activation='relu')(x1)
    x1 = MaxPooling3D(pool_size=(2, 2, 2))(x1)
    
    x1 = Flatten()(x1)
    x1 = Dense(500, activation='softmax')(x1)
    
    # creation des couches flow
    x2 = Conv3D(30, kernel_size=(3,3,3), padding='same', activation='relu')(flow_input)
    x2 = MaxPooling3D(pool_size=(2, 2, 2))(x2)
    
    x2 = Conv3D(60, kernel_size=(3,3,3), padding='same', activation='relu')(x2)
    x2 = MaxPooling3D(pool_size=(2, 2, 2))(x2)
    
    x2 = Conv3D(80, kernel_size=(3,3,3), padding='same', activation='relu')(x2)
    x2 = MaxPooling3D(pool_size=(2, 2, 2))(x2)
    
    x2 = Flatten()(x2)
    x2 = Dense(500, activation='softmax')(x2)
    
    # combinaison des deux branches
    x = Concatenate()([x1, x2])
    
    # full connection
    prediction = Dense(nb_class, activation='softmax')(x)
    
    # creation du modele
    model = Model(inputs=[rgb_input, flow_input], outputs=prediction)

    # compiler le modele
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    return model
    
