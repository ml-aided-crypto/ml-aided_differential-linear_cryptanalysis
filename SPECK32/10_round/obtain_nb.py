# -*- coding: utf-8 -*-

import numpy as np
from os import urandom
from math import log2
import speck as ciph
import gc


def input_difference():
    return (0x211,0xa04)


def output_difference():
    return (0x8100,0x8102)


def get_round():
    return 4


def hw(v):
    res = np.zeros(v.shape,dtype=np.uint8);
    for i in range(16):
        res = res + ((v >> i) & 1)
    return res

def nb_location(x):
    l=x[0]
    r=x[1]
    
    l=bin(l)[2:].zfill(16)
    r=bin(r)[2:].zfill(16)
    
    l=np.array(list(l),dtype=np.uint8)
    r=np.array(list(r),dtype=np.uint8)
    
    l=(15-np.where(l==1)[0])+16
    r=15-np.where(r==1)[0]
    
    return list(l)+list(r)


def difference_is_x(x):
    r=[]
    for i in range(x+1):
        r.append((i,x-i))
    diff=[]
    for i in r:
        low_weight = np.array(range(2**16), dtype=np.uint16)
        low_weight_l = np.array(low_weight[hw(low_weight) == i[0]], dtype=np.uint16) 
        low_weight_r = np.array(low_weight[hw(low_weight) == i[1]], dtype=np.uint16) 
        
        ll=len(low_weight_l)
        lr=len(low_weight_r)
        
        low_weight_l = np.tile(low_weight_l, lr) #
        low_weight_r = np.repeat(low_weight_r, ll)
        
        result=np.vstack((low_weight_l,low_weight_r))
        result=result.T
        result=result.tolist()
        diff=diff+result
    return diff


def obtain_probability():
    n=2**31
    Input=input_difference()
    Output=output_difference()
    nr=get_round()
    
    keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1)
    ks = ciph.expand_key(keys, nr)
    
    del keys
    gc.collect()
    
    plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16)
    plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16)
    plain1l = plain0l ^ Input[0]
    plain1r = plain0r ^ Input[1]

    ctdata0l, ctdata0r = ciph.encrypt((plain0l, plain0r), ks)
    ctdata1l, ctdata1r = ciph.encrypt((plain1l, plain1r), ks)
    
    del plain0l,plain0r,plain1l,plain1r
    gc.collect()
    
    difference_c=(np.array(ctdata0l^ctdata1l,dtype=np.uint32)<<16)+np.array(ctdata0r^ctdata1r,dtype=np.uint32)
    
    del ctdata0l,ctdata0r,ctdata1l,ctdata1r
    gc.collect()
    
    Output_difference=(Output[0]<<16)+Output[1]
    
    num=len(np.where(difference_c==Output_difference)[0])
    print(num/n)
    print(log2(num/n))
    
    return num/n,log2(num/n)
    

def find_conforming_pair():
    n=2**31
    Input=input_difference()
    Output=output_difference()
    nr=get_round()
    
    keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1)
    ks = ciph.expand_key(keys, nr)
    
    plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16)
    plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16)
    plain1l = plain0l ^ Input[0]
    plain1r = plain0r ^ Input[1]

    ctdata0l, ctdata0r = ciph.encrypt((plain0l, plain0r), ks)
    ctdata1l, ctdata1r = ciph.encrypt((plain1l, plain1r), ks)
    
    del plain1l,plain1r,ks
    gc.collect()
    
    difference_c=(np.array(ctdata0l^ctdata1l,dtype=np.uint32)<<16)+np.array(ctdata0r^ctdata1r,dtype=np.uint32)
    
    del ctdata0l,ctdata0r,ctdata1l,ctdata1r
    gc.collect()

    Output_difference=(Output[0]<<16)+Output[1]
    
    location=np.where(difference_c==Output_difference)[0]
    
    print('Rate of conforming pair: ',len(location)/n,log2(len(location)/n))
    
    return plain0l[location],plain0r[location],keys[:,location]

def find_NB():
    Input=input_difference()
    Output=output_difference()
    Output_difference=(Output[0]<<16)+Output[1]
    nr=get_round()

    hw_NB=3
    
    #
    NB=[]
    for i in range(1,hw_NB+1):
        NB=NB+difference_is_x(i)
    
    #obtain conforming_pair
    pl,pr,keys=find_conforming_pair()
    ks = ciph.expand_key(keys, nr)
    
    nb_probability=[]
    
    for nb in NB:
        pl_0=pl^nb[0]
        pr_0=pr^nb[1]
        
        pl_1=pl_0 ^ Input[0]
        pr_1=pr_0 ^ Input[1]
        
        ctdata0l, ctdata0r = ciph.encrypt((pl_0, pr_0), ks)
        ctdata1l, ctdata1r = ciph.encrypt((pl_1, pr_1), ks)
        
        difference_c=(np.array(ctdata0l^ctdata1l,dtype=np.uint32)<<16)+np.array(ctdata0r^ctdata1r,dtype=np.uint32)
        location=np.where(difference_c==Output_difference)[0]
        
        nb_probability.append(len(location)/len(pl))

        print(nb,nb_location(nb),len(location)/len(pl))
    
    np.save('NB.npy',np.array(NB))
    np.save('nb_probability.npy',np.array(nb_probability))
    
    return NB,nb_probability
    
    

def get_effective(a):
    flag=0
    all_NB=[]
    
    for i in range(pow(2,len(a))):
        use_NB=np.array(list(bin(i)[2:].zfill(len(a))),dtype=np.uint8)
        l=0
        r=0
        for location_nb in range(len(a)):
            
            if(use_NB[location_nb]==1):
                #print(NB[location_nb],end='*')
                l=l^a[location_nb][0]
                r=r^a[location_nb][1]
        all_NB.append(np.array([l,r]))
    
    for i in range(len(all_NB)-1):
        for j in range(i+1,len(all_NB)):
            if((all_NB[i][0]==all_NB[j][0]) and (all_NB[i][1]==all_NB[j][1])):
               flag=1
    return flag         
    


def find_nb(num):
    NB=np.load('NB.npy')
    nb_probability=np.load('nb_probability.npy')

    location=np.argsort(-nb_probability)#
    new_NB=NB[location]
    new_nb_probability=nb_probability[location]
    
    if(num==1):
        return [0]
    else:
        t1=find_nb(num-1)
    
        a=t1[-1]+1

        difference=[new_NB[i] for i in t1]

        for i in range(a,len(new_NB)):
            d1=[j for j in difference]
            d1.append(new_NB[i])

            if(get_effective(d1)==0):
                break
        t1.append(i)
        return t1
    
       
NB=np.load('NB.npy')
nb_probability=np.load('nb_probability.npy')

location=np.argsort(-nb_probability)#

new_NB=NB[location]
new_nb_probability=nb_probability[location]  

need_num_nb=8
t=find_nb(need_num_nb)

for i in t:
    print(i,new_NB[i],nb_location(new_NB[i]),new_nb_probability[i])
    
