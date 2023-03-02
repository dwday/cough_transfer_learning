# -*- coding: utf-8 -*-
"""

Transdferlearning models

"""
from tensorflow.keras.applications import MobileNet
from tensorflow.keras import layers,Model,losses
from tensorflow.keras import optimizers
#------------------------------------------------------------------------------


def modelMobileNet(rows=128,cols=128+256,lr=1e-2):
    base = MobileNet(include_top=False, input_shape=(rows,cols,3))        
    for layer in base.layers[:-3]:
        layer.trainable = False
    base.layers[2].trainable = True
    x=layers.AvgPool2D(2,2)(base.layers[-1].output)
    # flat1=layers.GlobalMaxPool2D()(base.layers[-3].output)
    # x=BatchNormalization()(base.layers[-1].output)
    x=layers.Flatten()(x) 
    x=layers.Dropout(0.5)(x)
    #x= Dense(32, activation='relu')(x)  
    output = layers.Dense(1, activation='sigmoid')(x)    
    model = Model(inputs=base.inputs,outputs=output)
    
    opt=optimizers.RMSprop(lr)
    # opt=optimizers.Adam(lr)
    model.compile(optimizer=opt,#Adam(lr=1e-2),
              loss=losses.binary_crossentropy,# losses.huber,
              metrics=['acc'])   
    model.summary()
    return model

