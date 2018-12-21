# -*- coding: utf-8 -*-
from __future__ import print_function
import time, random, datetime, gc
from src.functions import *
from src.model import *
from src.data import *


if __name__ == "__main__":

    data_type='Both'

    name_script = 'keras_%s_%s' % (data_type, datetime.datetime.now().strftime("%d-%m-%Y_%H-%M"))
    flow_normalization = get_flow_normalization(visualization = 1)
    class_index, train_list, test_list = define_list(train_file = '/net/ens/DeepLearning/lab5/Data_TP/list/trainlist.txt',
                                                     test_file = '/net/ens/DeepLearning/lab5/Data_TP/list/testlist.txt',
                                                     class_file ='/net/ens/DeepLearning/lab5/Data_TP/list/classInd.txt')

    width = 100; height = 100; temporal_dim = 100; nb_class = len(class_index); batch_size = 5; nb_epoch = 30
    random.seed(1)

    train_generator = My_Data_Sequence(train_list, flow_normalization, class_index, batch_size, augmentation=True)
    test_generator = My_Data_Sequence(test_list, flow_normalization, class_index, batch_size, shuffle=False)

    #1) create siamese model
    model = make_model(temporal_dim, width, height, nb_class)
    
    #2) load weights
    weights_file='/net/ens/DeepLearning/lab5/Data_TP/models/Siamese_model.hdf5'
    
    model.load_weights(weights_file)
    
    model.summary()
    
    score = model.evaluate_generator(test_generator)
    print('Score of our pre-trained model = ', score)

    log_dir = './TensorBoard/%s' % name_script; 
    #remove_file(log_dir); 

    # Callbacks
    tensorboard_call = keras.callbacks.TensorBoard(log_dir = log_dir, batch_size=batch_size, write_images=True)
    log_callback = keras.callbacks.CSVLogger(os.path.join('logs', '%s.log' % name_script), separator=',', append=False)
    checkpoint_weigths = keras.callbacks.ModelCheckpoint('models_saved/weigths_%s_epoch_{epoch:03d}-valloss_{val_loss:.3f}.hdf5' % name_script, save_weights_only=True)

    

    #3) train model
    print('Training :'); tic = time.time()
    
    model.fit_generator(train_generator, epochs=nb_epoch, verbose=1, validation_data=test_generator, use_multiprocessing=True)
    callbacks = [tensorboard_call, checkpoint_weigths, log_callback]
    
    score = model.evaluate_generator(test_generator)
    
    print('Score of our trained model = ', score)
    
    print('Training time : %.3g s' % (time.time() - tic))

    
