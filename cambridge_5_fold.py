#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

>>> Kindly, cite this work if you find it useful
@article{akgun2021transfer, title={A transfer learning-based deep learning approach for automated COVID-19diagnosis with audio data}, 
author={AKG{"U}N, DEVR{.I}M and KABAKU{\c{S}}, ABDULLAH TALHA and {\c{S}}ENT{"U}RK, ZEHRA KARAPINAR and {\c{S}}ENT{"U}RK, ARAFAT and K{"U}{\c{C}}{"U}KK{"U}LAHLI, ENVER}, 
journal={Turkish Journal of Electrical Engineering and Computer Sciences}, volume={29}, number={8}, pages={2807--2823}, year={2021} }

"""
import numpy as np 
import matplotlib.pyplot as plt
import models_file
from load_features import load_dataset_5_Fold
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
import os

epochs = 300

split  = 0.2#0.3
lr     = 1e-2
rows   = 128
cols   = 224#128+256
patien = 100 

# -- DATASET Directories ------------------------------------------------------
#PUT YOUR DIRECTORIES  HERE
dir_task1='/media/com/YEDEK/WORKSPACE/COVID19/cambridge-task-1/'

dir_covid=dir_task1+'covid/'
dir_healthy=dir_task1+'healthy/'


# -- LOAD DATASET -------------------------------------------------------------
if os.path.isfile('datasetcovid.npz'): 
    print('using previously recorded validation set ...')
    data=np.load('datasetcovid.npz')    
    train_data  = data['train_data']
    train_label = data['train_label']
else:
    print('extracting features ...')
    train_data,train_label=load_dataset_5_Fold(dir_covid,dir_healthy,rows,cols)


print('train_data.shape=',train_data.shape)

# class weight balancing ------------------------------------------------------
class_weights = class_weight.compute_class_weight(class_weight = 'balanced',
                                                  classes= np.unique(train_label),
                                                  y= train_label)
class_weights= {i : class_weights[i] for i in range(len(class_weights))}

print(class_weights)

# If dataset splitted previously, use it---------------------------------------
if os.path.isfile('data_kfs.npy'): 
    print('using previously recorded kfs ...')
    data=np.load('data_kfs.npy',allow_pickle=True)
    kfs=data.item()
else:
    kfs=StratifiedKFold(n_splits=5,random_state=42, shuffle=True)
    np.save('data_kfs.npy',kfs)
#------------------------------------------------------------------------------

 
def experiment(lr,batch_size=16):
    #------------------------------------------------------- ------------------
    tf.keras.backend.clear_session()
    Acc=np.zeros(5,)
    inx=0
    for train_index, test_index in kfs.split(train_data,train_label):
        
        earlyStopping = EarlyStopping(monitor='val_acc',#'val_loss',
                              patience=patien, 
                              verbose=0, 
                              mode='max')
        # modelCheckPoint = ModelCheckpoint('./models/model5class.weights.{epoch:03d}-{val_acc:.4f}.hdf5', 
        #                           save_best_only=True,
        #                           monitor='val_acc', 
        #                           mode='max')
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = train_data[train_index], train_data[test_index]
        y_train, y_test = train_label[train_index],train_label[test_index]
        
        
        from keras.preprocessing.image import ImageDataGenerator
        train_datagen = ImageDataGenerator( 
            #preprocessing_function=fonk, 
            # height_shift_range=0.3,
            zoom_range=0.2,
            horizontal_flip=True,
            # vertical_flip=True,
            fill_mode='reflect'
            )
        
        train_generator = train_datagen.flow(X_train,
                                             y_train,
                                             batch_size=batch_size)
      
        test_datagen = ImageDataGenerator()
        validation_generator  = test_datagen.flow(X_test,
                                                  y_test,
                                                  batch_size=batch_size)
        
                
        # model=models_file.modelVGG19(rows,cols,lr)
        model=models_file.modelMobileNet(rows,cols,lr)
        # model=models_file.modelResNet50V2(rows,cols,lr)
        #model=models_file.modelInceptionV3(rows,cols,lr)  
        # model=models_file.modelDenseNet121(rows,cols,lr)  
    
        #model.summary()
        #--------------------------------------------------------------------------    
       
        # print("X_train.shape=",X_train.shape,"X_test.shape=",X_test.shape)
        print("steps_per_epoch=",366//batch_size)
        print("validation_steps=",93//batch_size)
        history = model.fit_generator(
            train_generator,
            steps_per_epoch=366//batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=93//batch_size,
            callbacks=[earlyStopping],
            workers=3, verbose=0
            )
        
        #--------------------------------------------------------------------------    
        acc      = history.history['acc']
        val_acc  = history.history['val_acc']
        loss     = history.history['loss']
        val_loss = history.history['val_loss']
        #--------------------------------------------------------------------------
        import matplotlib.pyplot as plt
        max_inx     = np.argmax(val_acc)
        max_val_acc = val_acc[max_inx]
        Acc[inx]    = max_val_acc
        print('max_val_acc=',max_val_acc)
        inx=inx+1
        #sonuclar=max_val_acc
        num_iters=len(acc)
        
    print('Acc= ',Acc)   
    return Acc,num_iters



LearningRates = [5e-3,1e-3, 5e-4, 1e-4] 
BatchSize     = [4,   8   , 16  , 32 ,64 ]


AccN=np.zeros((len(BatchSize),len(LearningRates),5),dtype='float32')
NumOfIters=np.zeros((len(BatchSize),len(LearningRates)),dtype='int32')

import time
# t0= time.clock()
t0=time.process_time()
Acc=[]
for b in range(len(BatchSize)):
    for i in range(len(LearningRates)):
        print("batch=",BatchSize[b],"lr=",LearningRates[i])
        #Acc.append(experiment(LearningRates[i],BatchSize[b]))
        max_val_acc,num_iters = experiment(LearningRates[i],BatchSize[b])
        AccN[b,i,:]             = max_val_acc
        NumOfIters[b,i]       = num_iters

t1 = time.process_time() - t0
print("Time elapsed: ", t1) # CPU seconds elapsed (floating point)

#for i in range(len(LearningRates)):
#    print('lr=',LearningRates[i], 'Max Acc=',Acc[i])


for b in range(len(BatchSize)):
    for i in range(len(LearningRates)):
        print('Batch size=',BatchSize[b],
              '  Learning rate=',LearningRates[i],
              ' Acc=',AccN[b,i,:],'Average=',np.mean(AccN[b,i,:]),' num_iters=',NumOfIters[b,i]) 



"""
steps_per_epoch= 5
validation_steps= 1
max_val_acc= 0.890625
Acc=  [0.890625 0.9375   0.890625 0.875    0.890625]
Time elapsed:  52322.061112922005
Batch size= 4   Learning rate= 0.005  Acc= [0.8695652  0.8913044  0.82417583 0.84615386 0.83516484] Average= 0.8532728  num_iters= 184
Batch size= 4   Learning rate= 0.001  Acc= [0.8695652  0.9130435  0.82417583 0.83516484 0.82417583] Average= 0.85322505  num_iters= 149
Batch size= 4   Learning rate= 0.0005  Acc= [0.8804348  0.90217394 0.84615386 0.85714287 0.82417583] Average= 0.8620163  num_iters= 109
Batch size= 4   Learning rate= 0.0001  Acc= [0.8804348  0.9130435  0.83516484 0.82417583 0.84615386] Average= 0.8597945  num_iters= 146
Batch size= 8   Learning rate= 0.005  Acc= [0.8863636  0.90909094 0.84090906 0.85227275 0.8863636 ] Average= 0.875  num_iters= 300
Batch size= 8   Learning rate= 0.001  Acc= [0.875      0.89772725 0.82954544 0.82954544 0.85227275] Average= 0.8568182  num_iters= 187
Batch size= 8   Learning rate= 0.0005  Acc= [0.89772725 0.90909094 0.84090906 0.84090906 0.85227275] Average= 0.8681818  num_iters= 224
Batch size= 8   Learning rate= 0.0001  Acc= [0.89772725 0.89772725 0.85227275 0.85227275 0.8636364 ] Average= 0.8727273  num_iters= 110
Batch size= 16   Learning rate= 0.005  Acc= [0.8875 0.9125 0.85   0.8625 0.875 ] Average= 0.87749994  num_iters= 295
Batch size= 16   Learning rate= 0.001  Acc= [0.8875 0.9125 0.85   0.8375 0.8375] Average= 0.86500007  num_iters= 158
Batch size= 16   Learning rate= 0.0005  Acc= [0.8625 0.925  0.8625 0.85   0.8375] Average= 0.8675  num_iters= 126
Batch size= 16   Learning rate= 0.0001  Acc= [0.9    0.925  0.85   0.85   0.8625] Average= 0.87750006  num_iters= 227
Batch size= 32   Learning rate= 0.005  Acc= [0.90625  0.90625  0.890625 0.875    0.859375] Average= 0.8875  num_iters= 160
Batch size= 32   Learning rate= 0.001  Acc= [0.890625 0.9375   0.875    0.859375 0.859375] Average= 0.884375  num_iters= 176
Batch size= 32   Learning rate= 0.0005  Acc= [0.875    0.9375   0.875    0.875    0.890625] Average= 0.890625  num_iters= 184
Batch size= 32   Learning rate= 0.0001  Acc= [0.90625 0.9375  0.875   0.84375 0.875  ] Average= 0.8875  num_iters= 139
Batch size= 64   Learning rate= 0.005  Acc= [0.875    0.921875 0.90625  0.859375 0.890625] Average= 0.890625  num_iters= 205
Batch size= 64   Learning rate= 0.001  Acc= [0.90625  0.90625  0.90625  0.828125 0.890625] Average= 0.8875  num_iters= 218
Batch size= 64   Learning rate= 0.0005  Acc= [0.90625  0.921875 0.90625  0.828125 0.859375] Average= 0.884375  num_iters= 208
Batch size= 64   Learning rate= 0.0001  Acc= [0.890625 0.9375   0.890625 0.875    0.890625] Average= 0.896875  num_iters= 300
"""
