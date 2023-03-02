# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 08:04:34 2021

@author: bilg
"""
import librosa 
import librosa.display
import numpy as np
import os
from keras.preprocessing import image
from sklearn.model_selection import train_test_split

def feature_extract(fname):
    feat =0
    try:
        print(fname)
        sr=librosa.get_samplerate(fname)        
        audio,sr = librosa.load(fname,sr=sr)   

        print('sr=',sr,)
        print(' get_samplerate(fname) =',librosa.get_samplerate(fname))
        print('audio.shape=',audio.shape)
        # audio=audio[np.nonzero(audio>0) ]
        print('audio.shape=',audio.shape)
        
        # audio=audio/audio.std()
        # audio= (audio-audio.min())/(audio.max()-audio.min())
        # print('File loaded')
        feat= librosa.feature.melspectrogram(y=audio,sr=sr)
        feat   = librosa.power_to_db(feat, ref=np.max) 
        # feat= feat-np.min(feat)
        # feat=np.log(feat+1)
        feat= (feat-feat.min())/(feat.max()-feat.min())
    except:
        print('File cannot open')  
        
        
    return feat

def load_features(dir_covid,dir_healthy):    
    #------------------------------------------------------------------------------
    #train data
    dir_content_covid   = os.listdir(dir_covid)
    #test data
    dir_content_healthy = os.listdir(dir_healthy)
    data_train=[]
    max_len=0
    sy=0
    for fname in dir_content_covid:
        if fname[-3:] =='wav':            
            try:            
                f=feature_extract(dir_covid+fname)
                
                if f.max()>0:
                    data_train.append(f)
                    print('f.shape2=',f.shape)
                    if max_len<f.shape[1]:
                        max_len=f.shape[1]
                else:
                   print('error covid: ')
                   print(fname)
                   sy=sy+1
                   
                #print(f.shape)
            except:
                print('error covid: ')
                print(fname)
                
    s1=len(data_train)
    #------------------------------------------------------------------------------
    for fname in dir_content_healthy:
        if fname[-3:] =='wav':
            try: 
                #print(fname)
                f=feature_extract(dir_healthy+fname)
                if f.max()>0:
                    data_train.append(f)
                    print('f.shape2=',f.shape)
                    if max_len<f.shape[1]:
                        max_len=f.shape[1]
                else:
                    print('error healty: ')
                    print(fname)
                    sy=sy+1
                #print(f.shape)
            except:
                print('error healty: ')
                print(fname)
    s=len(data_train)
    #------------------------------------------------------------------------------
    data_label=np.zeros(s,)
    data_label[0:s1]=np.ones(s1,)
    #------------------------------------------------------------------------------
    print('num  of files  not opened : ',sy)
    print('max_len=',max_len)
    return (data_train,data_label)

    #return (train_data,train_label),(test_data,test_label)




def imresize(data_train,rows,cols):
    s=len(data_train)
    data_trainX=np.zeros((s,rows,cols),dtype='float32')
    
    # ** RESIZE to WxH
    for i in range(0,s):
        #print('data_train[i].shape=',data_train[i].shape)
        shp=data_train[i].shape
        if shp[1]>cols:
            data_trainX[i,:,:]=image.smart_resize(
                data_train[i].reshape(data_train[i].shape[0],
                                      data_train[i].shape[1],1),
                (rows,cols)).reshape(rows,cols)
        else:
            R=image.smart_resize(
            data_train[i].reshape(data_train[i].shape[0],
                                  data_train[i].shape[1],1),
            (rows,data_train[i].shape[1])).reshape(rows,data_train[i].shape[1])
            
            img=np.zeros((rows,cols),dtype='float')
            B=np.asarray(R)
            img[:B.shape[0],:B.shape[1]]=B
            data_trainX[i,:,:]=img
        
    return data_trainX


def load_dataset_5_Fold(dir_covid,dir_healthy,rows,cols):
    
    # Extract features
    (data_trainF,train_label)=load_features(dir_covid,dir_healthy)
    
    # resize to WxH
    print('** image resize to ', rows,'x',cols)
    data_trainX=imresize(data_trainF,rows,cols)
    
    b=data_trainX.shape
    train_data=np.zeros((b[0],b[1],b[2],3))
    train_data[:,:,:,0]=data_trainX
    train_data[:,:,:,1]=data_trainX
    train_data[:,:,:,2]=data_trainX 
    
    train_data=np.nan_to_num(train_data)
    
    print('train_data.shape=',train_data.shape)
  
    np.savez('datasetcovid.npz',train_data=train_data,train_label=train_label)

    return train_data,train_label


def load_dataset(dir_covid,dir_healthy,rows,cols,split=0.2):
    
    # Extract features
    (data_trainF,data_labelF)=load_features(dir_covid,dir_healthy)
    
    # resize to WxH
    print('** image resize to ', rows,'x',cols)
    data_trainX=imresize(data_trainF,rows,cols)
    
  
    # ** TRAIN-VAL SPLIT    
    train_data, val_data, train_label , val_label=\
        train_test_split(data_trainX, 
                         data_labelF, 
                         test_size=split,
                         stratify=data_labelF,
                         shuffle=True) 
    
    
    train_data=np.nan_to_num(train_data)
    val_data=np.nan_to_num(val_data)
    
    #train_data=train_data*255.0
    #val_data=val_data*255.0
    
    # ** RGB repeat
    b=train_data.shape
    train_dataX=np.zeros((b[0],b[1],b[2],3))
    train_dataX[:,:,:,0]=train_data
    train_dataX[:,:,:,1]=train_data
    train_dataX[:,:,:,2]=train_data
    
    b=val_data.shape
    val_dataX=np.zeros((b[0],b[1],b[2],3))
    val_dataX[:,:,:,0]=val_data
    val_dataX[:,:,:,1]=val_data
    val_dataX[:,:,:,2]=val_data
    
    train_data=train_dataX
    val_data=val_dataX
    
    print('len(train_data)=',len(train_data))
    print('len(val_data  )=',len(val_data))
    
    np.savez('datasetcovid.npz',train_data=train_data,train_label=train_label,val_data=val_data,val_label=val_label)

    return train_data,train_label,val_data,val_label