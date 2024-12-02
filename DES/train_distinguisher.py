import numpy as np
from os import urandom
import gc 
from multiprocessing import Process,Queue,Pool
from time import time

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pickle
import tensorflow as tf  
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)  
config.gpu_options.allow_growth=True  
session = tf.compat.v1.Session(config=config)  
tf.compat.v1.keras.backend.set_session(session)#
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2


import des_for_linear as ciph
#only for 6-round des


def get_active_bit():
    return [1,2]

def get_Round():
    return 6

def obtain_probability_expression():
    #
    num=2**20

    keys = np.frombuffer(urandom(64), dtype=np.uint8).reshape(-1,64)
    keys = keys & 1
    subkey=ciph.expand_key(keys,get_Round())#

    plain = np.frombuffer(urandom(64*num), dtype=np.uint8).reshape(-1,64)
    plain = plain & 1
    
    plain1=np.array(plain)
    for i in get_active_bit():
        plain1[:,i]=plain1[:,i]^np.ones(len(plain1[:,i]),dtype=plain1[:,i][0].dtype)

    Ph=np.array(plain[:,:32])
    Pl=np.array(plain[:,32:])
    Ph1=np.array(plain1[:,:32])
    Pl1=np.array(plain1[:,32:])
    
    del plain,plain1
    gc.collect()

    for sk in subkey:
        Ph,Pl = ciph.enc_one_round((Ph,Pl), sk)
        Ph1,Pl1 = ciph.enc_one_round((Ph1,Pl1), sk)
        
    cipher=np.concatenate((Pl,Ph),axis=1)
    cipher1=np.concatenate((Pl1,Ph1),axis=1)

    Ch=np.array(cipher[:,:32])
    Cl=np.array(cipher[:,32:])
    
    Ch1=np.array(cipher1[:,:32])
    Cl1=np.array(cipher1[:,32:])

    value=Ch[:,31-7]^Ch[:,31-18]^Ch[:,31-24]^Ch[:,31-29]^Cl[:,31-15]^Ch1[:,31-7]^Ch1[:,31-18]^Ch1[:,31-24]^Ch1[:,31-29]^Cl1[:,31-15]
    
    value=np.array(value,dtype=np.uint64)
    
    
    return abs(0.5-sum(value)/num)

def create_model(num_pair):
    num_filters=2
    num_outputs=1
    d1=num_pair
    d2=num_pair

    ks=3
    reg_param=0.0001
    inp = Input(shape=(num_pair,))
    rs = Reshape((1, num_pair))(inp)
    #perm = Permute((2,1))(rs)
    #add a single residual layer that will expand the data to num_filters channels
    #this is a bit-sliced layer
    conv0 = Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))(rs)
    conv0 = BatchNormalization()(conv0)
    conv0 = Activation('relu')(conv0)
    #add residual blocks
    shortcut = conv0
    for i in range(5):
        conv1 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(shortcut)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv2 = Conv1D(num_filters, kernel_size=ks, padding='same',kernel_regularizer=l2(reg_param))(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        shortcut = Add()([shortcut, conv2])
    #add prediction head
    flat1 = Flatten()(shortcut)
    dense1 = Dense(d1,kernel_regularizer=l2(reg_param))(flat1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)
    out = Dense(num_outputs, activation='sigmoid', kernel_regularizer=l2(reg_param))(dense2)
    model = Model(inputs=inp, outputs=out)
    return(model)

def make_train_data(n,num_pair):
    
    Round=get_Round()
    
    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    Y = Y & 1
    
    X = np.frombuffer(urandom(num_pair*n), dtype=np.uint8).reshape(-1,num_pair)
    X = X & 1
    
    cipher_X=[]
    keys = np.frombuffer(urandom(64*np.sum(Y==1)),dtype=np.uint8).reshape(-1,64)
    keys = keys & 1
    subkey=ciph.expand_key(keys,Round)
    
    for j in range(num_pair):
        
        plain = np.frombuffer(urandom(64*np.sum(Y==1)), dtype=np.uint8).reshape(-1,64)
        plain = plain & 1
        
        plain1=np.array(plain)
        for i in get_active_bit():
            plain1[:,i]=plain1[:,i]^np.ones(len(plain1[:,i]),dtype=plain1[:,i][0].dtype)

        Ph=np.array(plain[:,:32])
        Pl=np.array(plain[:,32:])
        Ph1=np.array(plain1[:,:32])
        Pl1=np.array(plain1[:,32:])

        
        for sk in subkey:
            Ph,Pl = ciph.enc_one_round((Ph,Pl), sk)
            Ph1,Pl1 = ciph.enc_one_round((Ph1,Pl1), sk)
            
        cipher=np.concatenate((Pl,Ph),axis=1)
        cipher1=np.concatenate((Pl1,Ph1),axis=1)

        Ch=np.array(cipher[:,:32])
        Cl=np.array(cipher[:,32:])
        
        Ch1=np.array(cipher1[:,:32])
        Cl1=np.array(cipher1[:,32:])

        value=Ch[:,31-7]^Ch[:,31-18]^Ch[:,31-24]^Ch[:,31-29]^Cl[:,31-15]^Ch1[:,31-7]^Ch1[:,31-18]^Ch1[:,31-24]^Ch1[:,31-29]^Cl1[:,31-15]
        

        cipher_X.append(value)
    
    cipher_X=np.array(cipher_X,dtype=np.uint8)
    cipher_X=cipher_X.T
    

    X[Y==1]=cipher_X
    
    return X,Y


def train_model(num_pair):
    net_name='DES_DL_'+str(get_Round())+'_'+str(num_pair)
    
    train_data_size=2**25
    train_data,train_flag=make_train_data(train_data_size,num_pair)
    val_data,val_flag=make_train_data(int(train_data_size/8),num_pair)
    
    seed=3407
    np.random.seed(seed)
    model=create_model(num_pair)
    model.compile(optimizer='adam',loss='mse',metrics=['acc'])
    filepath_net=net_name+'_weight'+'.h5'
    checkpoint=ModelCheckpoint(filepath=filepath_net,monitor='val_acc',verbose=1,save_best_only=True,mode='max')
    callback_list=[checkpoint]
    history=model.fit(train_data,train_flag,validation_data=(val_data,val_flag),epochs=50,batch_size=2**17,verbose=1,callbacks=callback_list)
    with open(net_name+'.txt','wb') as file:        
        pickle.dump(history.history,file)
    model_json=model.to_json()
    with open(net_name+'_model'+'.json','w') as file:
        file.write(model_json)
    return max(np.array(history.history['val_acc']))

result=[]

Num_pair=[16,18,20]

for i in Num_pair:
    acc=train_model(i)
    result.append([i,acc])
    print(result)
