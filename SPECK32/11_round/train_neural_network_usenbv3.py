# -*- coding: utf-8 -*-


import speck as ciph
import numpy as np
from os import urandom
from time import time
import gc


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"#
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
from tensorflow.keras.models import model_from_json


def get_bit_value(num,x):
    if(x>0):
        num=num>>x
    return np.array(num-((num>>1)<<1),dtype=np.uint8)


def get_Round():
    return 9


def DIFF():
    return (0x211,0xa04)


def MASK():
    l_mask=0x5820
    r_mask=0x4020

    a=np.array(list(bin(l_mask)[2:].zfill(16)),dtype=np.uint8)
    
    b=np.array(list(bin(r_mask)[2:].zfill(16)),dtype=np.uint8)
    
    a=np.where(a==1)[0]
    b=np.where(b==1)[0]
    
    return (15-a,15-b)

def get_neural_bits(n):
    
    use_NB=[np.array([0,0])]
    
    NB=np.load('NB.npy')
    nb_probability=np.load('nb_probability.npy')

    location=np.argsort(-nb_probability)#

    new_NB=NB[location]

    for i in range(n-1):
        use_NB.append(new_NB[i])
    return use_NB
    
    
    

def make_one_key_data():

    nr=get_Round()
    l_mask,r_mask=MASK()
    RES=np.zeros(2**32,dtype=np.uint8)
    
    diff=DIFF()
    
    keys = np.frombuffer(urandom(8),dtype=np.uint16).reshape(4,-1)
    
    plain0l = np.arange(2**16,dtype=np.uint16)
    plain0r = np.arange(2**16,dtype=np.uint16)
    
    plain0l=plain0l.repeat(2**16)
    plain0r=np.tile(plain0r,2**16)
    
    plain1l = plain0l ^ diff[0]
    plain1r = plain0r ^ diff[1]

    ks = ciph.expand_key(keys, nr)

    ctdata0l, ctdata0r = ciph.encrypt((plain0l, plain0r), ks)
    
    if(len(l_mask)>0):
        for i in l_mask:
            RES=RES^get_bit_value(ctdata0l,i)
    if(len(r_mask)>0):
        for i in r_mask:
            RES=RES^get_bit_value(ctdata0r,i)
            

    del plain0l, plain0r,ctdata0l, ctdata0r
    gc.collect()
    
    ctdata1l, ctdata1r = ciph.encrypt((plain1l, plain1r), ks)
    del plain1l, plain1r
    gc.collect()
    
    if(len(l_mask)>0):
        for i in l_mask:

            RES=RES^get_bit_value(ctdata1l,i)
    if(len(r_mask)>0):
        for i in r_mask:

            RES=RES^get_bit_value(ctdata1r,i)
    
    
    RES=np.array(RES,dtype=np.float64)
    
    result=2*(sum(RES==0)/2**32)-1
    
    return result


def experiment_correlation():
    n=100
    S=0
    for i in range(n):
        S=S+make_one_key_data()

    return S/n



def make_train_data(n,num_pair):
    
    NB=get_neural_bits(num_pair)

    
    diff=DIFF()
    l_mask,r_mask=MASK()
    Round=get_Round()
    
    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    Y = Y & 1
    
    X = np.frombuffer(urandom(num_pair*n), dtype=np.uint8).reshape(-1,num_pair)
    X = X & 1
    
    cipher_X=[]
    keys = np.frombuffer(urandom(8*np.sum(Y==1)),dtype=np.uint16).reshape(4,-1)
    subkey=ciph.expand_key(keys,Round)
    
    p0l = np.frombuffer(urandom(2*np.sum(Y==1)),dtype=np.uint16)
    p0r = np.frombuffer(urandom(2*np.sum(Y==1)),dtype=np.uint16)
    #p1l = p0l ^ diff[0]
    #p1r = p0r ^ diff[1]

    for j in range(num_pair):
        
        use_nb=NB[j]
        print(use_nb)
        plain0l=np.array(p0l)^use_nb[0]
        plain0r=np.array(p0r)^use_nb[1]
        
        plain1l = plain0l ^ diff[0]
        plain1r = plain0r ^ diff[1]       
        
        
        ctdata0l, ctdata0r = ciph.encrypt((plain0l, plain0r), subkey)
        ctdata1l, ctdata1r = ciph.encrypt((plain1l, plain1r), subkey)
        
        RES=np.zeros(np.sum(Y==1),dtype=np.uint8)
        
        if(len(l_mask)>0):
            for i in l_mask:
                RES=RES^get_bit_value(ctdata0l,i)
                RES=RES^get_bit_value(ctdata1l,i)
        if(len(r_mask)>0):
            for i in r_mask:
                RES=RES^get_bit_value(ctdata0r,i)
                RES=RES^get_bit_value(ctdata1r,i)
        
        cipher_X.append(RES)
    
    cipher_X=np.array(cipher_X,dtype=np.uint8)
    cipher_X=cipher_X.T
    

    X[Y==1]=cipher_X
    
    return X,Y


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
    

def train_model(num_pair):
    net_name='SPECK32_DL_NBv3_'+str(get_Round())+'_'+str(num_pair)
    
    train_data_size=2**26
    train_data,train_flag=make_train_data(train_data_size,num_pair)
    val_data,val_flag=make_train_data(int(train_data_size/8),num_pair)
    
    seed=3407
    np.random.seed(seed)
    model=create_model(num_pair)
    model.compile(optimizer='adam',loss='mse',metrics=['acc'])
    filepath_net=net_name+'_weight'+'.h5'
    checkpoint=ModelCheckpoint(filepath=filepath_net,monitor='val_acc',verbose=1,save_best_only=True,mode='max')
    callback_list=[checkpoint]
    history=model.fit(train_data,train_flag,validation_data=(val_data,val_flag),epochs=100,batch_size=2**17,verbose=1,callbacks=callback_list)
    with open(net_name+'.txt','wb') as file:        
        pickle.dump(history.history,file)
    model_json=model.to_json()
    with open(net_name+'_model'+'.json','w') as file:
        file.write(model_json)
    return max(np.array(history.history['val_acc']))

result=[]

Num_pair=[512,1024]

for i in Num_pair:
    acc=train_model(i)
    result.append([i,acc])
    print(result)
    
    
 
    
    
    
    
    
