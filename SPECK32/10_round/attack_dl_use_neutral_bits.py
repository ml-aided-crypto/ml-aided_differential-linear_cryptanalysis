# -*- coding: utf-8 -*-


import numpy as np
from os import urandom

import speck as ciph
import gc
from time import time


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"#
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
from math import log2

def get_Round():
    return 8

def get_pc_pair():
    return 256


def get_bit_value(num,x):
    if(x>0):
        num=num>>x
    return np.array(num-((num>>1)<<1),dtype=np.uint8)


def DIFF():
    return (0x211,0xa04)


def MASK():
    l_mask=0x0008
    r_mask=0x0008

    a=np.array(list(bin(l_mask)[2:].zfill(16)),dtype=np.uint8)
    
    b=np.array(list(bin(r_mask)[2:].zfill(16)),dtype=np.uint8)
    
    a=np.where(a==1)[0]
    b=np.where(b==1)[0]
    
    return (15-a,15-b)

def get_neural_bits():
    
    use_NB=[np.array([0,0])]
    
    NB=np.load('NB.npy')
    nb_probability=np.load('nb_probability.npy')

    location=np.argsort(-nb_probability)#

    new_NB=NB[location]

    for i in range(255):
        use_NB.append(new_NB[i])
    return np.array(use_NB)

def make_testset(number):
    
    NB=get_neural_bits()
    NB_l=np.array(NB[:,0],dtype=np.uint16)
    NB_r=np.array(NB[:,1],dtype=np.uint16)
    NB_l=np.tile(NB_l,number)
    NB_r=np.tile(NB_r,number)

    Round=get_Round()+1
    diff=DIFF()
    
    keys = np.frombuffer(urandom(8),dtype=np.uint16).reshape(4,-1)
    
    subkey=ciph.expand_key(keys,Round)
    
    plain0l = np.frombuffer(urandom(2*number),dtype=np.uint16)
    plain0l=np.repeat(plain0l, len(NB))
    
    plain0r = np.frombuffer(urandom(2*number),dtype=np.uint16)
    plain0r=np.repeat(plain0r, len(NB))
    
    plain0l=plain0l^NB_l
    plain0r=plain0r^NB_r
    
    plain1l = plain0l ^ diff[0]
    plain1r = plain0r ^ diff[1]
    
    ctdata0l, ctdata0r = ciph.encrypt((plain0l, plain0r), subkey)
    ctdata1l, ctdata1r = ciph.encrypt((plain1l, plain1r), subkey)
    
    return ctdata0l, ctdata0r,ctdata1l, ctdata1r,subkey[-1]
    #return plain0l, plain0r


def make_one_attack():
    l_mask,r_mask=MASK()
    subkey_bits=13#
    net_json_file = open('SPECK32_DL_NBv3_'+str(get_Round())+'_'+str(get_pc_pair())+'_model.json','r')
    net_json_model = net_json_file.read()
    net = model_from_json(net_json_model)
    net.load_weights('SPECK32_DL_NBv3_'+str(get_Round())+'_'+str(get_pc_pair())+'_weight.h5')
    
    number=10000
    ct0l, ct0r, ct1l, ct1r, real_subkey = make_testset(number)
    
    key_score=[]
    number_key=2**subkey_bits
    for guess_key in range(number_key):
        
        RES=np.zeros(get_pc_pair()*number,dtype=np.uint8)

        pt0r=ciph.ror(ct0l ^ ct0r, ciph.BETA())
        pt0l= (((ct0l ^ guess_key) & (2**subkey_bits-1))-(pt0r & (2**subkey_bits-1))) & (2**subkey_bits-1)
        pt0l=np.array(pt0l>>12,dtype=np.uint8)
        
        pt1r=ciph.ror(ct1l ^ ct1r, ciph.BETA())
        pt1l= (((ct1l ^ guess_key) & (2**subkey_bits-1))-(pt1r & (2**subkey_bits-1))) & (2**subkey_bits-1)
        pt1l=np.array(pt1l>>12,dtype=np.uint8)

        if(len(r_mask)>0):
            for i in r_mask:
                RES=RES^get_bit_value(pt1r,i)
                RES=RES^get_bit_value(pt0r,i)
        
        RES=RES^pt0l^pt1l
        del pt0l,pt0r,pt1l,pt1r
        gc.collect()

        RES=RES.reshape(-1,get_pc_pair())
        result=net.predict(RES,batch_size=2**18)
        result=result.flatten()
        result=result/(1-result)
        result = np.log2(result)
        #print((real_subkey&(2**subkey_bits-1))[0],guess_key,sum(result))
        
        key_score.append(sum(result))
    
    key_score=np.array(key_score)
    guess_key=np.where(key_score==max(key_score))[0][0]
    
    realkey=(real_subkey&(2**subkey_bits-1))[0]

    aaa=key_score[realkey]
    
    
    index=len(np.where(key_score>aaa)[0])
    

    return guess_key,realkey,index,key_score


num=100
flag=0

name=3

for i in range(num): 
    t0=time()
    g,r,index,key_score=make_one_attack()
    np.save('key_score' + str(i)+'.npy', key_score)
    t1=time()
    filename ='result.txt'
    with open(filename,'a') as f: # 
        f.write(str(i)+'---'+str(g)+'---'+str(r)+'---'+str(index)+"\n")
        
    
    if(r==g):
        flag=flag+1
    print(i,bin(g^r)[2:],flag/(i+1),index,t1-t0)

print(flag/num)
        


