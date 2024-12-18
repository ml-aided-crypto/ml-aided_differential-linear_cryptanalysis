# -*- coding: utf-8 -*-

import numpy as np
from os import urandom

from math import log2

import des_for_linear as ciph


def num2bitarray6(x):
    r=bin(x)[2:]
    r=r.zfill(6)
    r=list(r)
    r=np.array(r,dtype=np.uint8)
    return r

def get_active_bit():
    return [1,2]

def bit2num(x):
    return (x[:,0]<<5)+(x[:,1]<<4)+(x[:,2]<<3)+(x[:,3]<<2)+(x[:,4]<<1)+(x[:,5]<<0)


def get_Round():
    return 6


def S_box():
    s0=np.array([[1,1,1,0],[0,0,0,0],[0,1,0,0],[1,1,1,1],[1,1,0,1],[0,1,1,1],[0,0,0,1],[0,1,0,0],[0,0,1,0],[1,1,1,0],[1,1,1,1],[0,0,1,0],[1,0,1,1],[1,1,0,1],[1,0,0,0],[0,0,0,1],[0,0,1,1],[1,0,1,0],[1,0,1,0],[0,1,1,0],[0,1,1,0],[1,1,0,0],[1,1,0,0],[1,0,1,1],[0,1,0,1],[1,0,0,1],[1,0,0,1],[0,1,0,1],[0,0,0,0],[0,0,1,1],[0,1,1,1],[1,0,0,0],[0,1,0,0],[1,1,1,1],[0,0,0,1],[1,1,0,0],[1,1,1,0],[1,0,0,0],[1,0,0,0],[0,0,1,0],[1,1,0,1],[0,1,0,0],[0,1,1,0],[1,0,0,1],[0,0,1,0],[0,0,0,1],[1,0,1,1],[0,1,1,1],[1,1,1,1],[0,1,0,1],[1,1,0,0],[1,0,1,1],[1,0,0,1],[0,0,1,1],[0,1,1,1],[1,1,1,0],[0,0,1,1],[1,0,1,0],[1,0,1,0],[0,0,0,0],[0,1,0,1],[0,1,1,0],[0,0,0,0],[1,1,0,1]],dtype=np.uint8)
    s1=np.array([[1,1,1,1],[0,0,1,1],[0,0,0,1],[1,1,0,1],[1,0,0,0],[0,1,0,0],[1,1,1,0],[0,1,1,1],[0,1,1,0],[1,1,1,1],[1,0,1,1],[0,0,1,0],[0,0,1,1],[1,0,0,0],[0,1,0,0],[1,1,1,0],[1,0,0,1],[1,1,0,0],[0,1,1,1],[0,0,0,0],[0,0,1,0],[0,0,0,1],[1,1,0,1],[1,0,1,0],[1,1,0,0],[0,1,1,0],[0,0,0,0],[1,0,0,1],[0,1,0,1],[1,0,1,1],[1,0,1,0],[0,1,0,1],[0,0,0,0],[1,1,0,1],[1,1,1,0],[1,0,0,0],[0,1,1,1],[1,0,1,0],[1,0,1,1],[0,0,0,1],[1,0,1,0],[0,0,1,1],[0,1,0,0],[1,1,1,1],[1,1,0,1],[0,1,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,1],[1,0,1,1],[1,0,0,0],[0,1,1,0],[1,1,0,0],[0,1,1,1],[0,1,1,0],[1,1,0,0],[1,0,0,1],[0,0,0,0],[0,0,1,1],[0,1,0,1],[0,0,1,0],[1,1,1,0],[1,1,1,1],[1,0,0,1]],dtype=np.uint8)
    s2=np.array([[1,0,1,0],[1,1,0,1],[0,0,0,0],[0,1,1,1],[1,0,0,1],[0,0,0,0],[1,1,1,0],[1,0,0,1],[0,1,1,0],[0,0,1,1],[0,0,1,1],[0,1,0,0],[1,1,1,1],[0,1,1,0],[0,1,0,1],[1,0,1,0],[0,0,0,1],[0,0,1,0],[1,1,0,1],[1,0,0,0],[1,1,0,0],[0,1,0,1],[0,1,1,1],[1,1,1,0],[1,0,1,1],[1,1,0,0],[0,1,0,0],[1,0,1,1],[0,0,1,0],[1,1,1,1],[1,0,0,0],[0,0,0,1],[1,1,0,1],[0,0,0,1],[0,1,1,0],[1,0,1,0],[0,1,0,0],[1,1,0,1],[1,0,0,1],[0,0,0,0],[1,0,0,0],[0,1,1,0],[1,1,1,1],[1,0,0,1],[0,0,1,1],[1,0,0,0],[0,0,0,0],[0,1,1,1],[1,0,1,1],[0,1,0,0],[0,0,0,1],[1,1,1,1],[0,0,1,0],[1,1,1,0],[1,1,0,0],[0,0,1,1],[0,1,0,1],[1,0,1,1],[1,0,1,0],[0,1,0,1],[1,1,1,0],[0,0,1,0],[0,1,1,1],[1,1,0,0]],dtype=np.uint8)
    s3=np.array([[0,1,1,1],[1,1,0,1],[1,1,0,1],[1,0,0,0],[1,1,1,0],[1,0,1,1],[0,0,1,1],[0,1,0,1],[0,0,0,0],[0,1,1,0],[0,1,1,0],[1,1,1,1],[1,0,0,1],[0,0,0,0],[1,0,1,0],[0,0,1,1],[0,0,0,1],[0,1,0,0],[0,0,1,0],[0,1,1,1],[1,0,0,0],[0,0,1,0],[0,1,0,1],[1,1,0,0],[1,0,1,1],[0,0,0,1],[1,1,0,0],[1,0,1,0],[0,1,0,0],[1,1,1,0],[1,1,1,1],[1,0,0,1],[1,0,1,0],[0,0,1,1],[0,1,1,0],[1,1,1,1],[1,0,0,1],[0,0,0,0],[0,0,0,0],[0,1,1,0],[1,1,0,0],[1,0,1,0],[1,0,1,1],[0,0,0,1],[0,1,1,1],[1,1,0,1],[1,1,0,1],[1,0,0,0],[1,1,1,1],[1,0,0,1],[0,0,0,1],[0,1,0,0],[0,0,1,1],[0,1,0,1],[1,1,1,0],[1,0,1,1],[0,1,0,1],[1,1,0,0],[0,0,1,0],[0,1,1,1],[1,0,0,0],[0,0,1,0],[0,1,0,0],[1,1,1,0]],dtype=np.uint8)
    s4=np.array([[0,0,1,0],[1,1,1,0],[1,1,0,0],[1,0,1,1],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,1,0,0],[0,1,1,1],[0,1,0,0],[1,0,1,0],[0,1,1,1],[1,0,1,1],[1,1,0,1],[0,1,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,1],[0,1,0,1],[0,0,0,0],[0,0,1,1],[1,1,1,1],[1,1,1,1],[1,0,1,0],[1,1,0,1],[0,0,1,1],[0,0,0,0],[1,0,0,1],[1,1,1,0],[1,0,0,0],[1,0,0,1],[0,1,1,0],[0,1,0,0],[1,0,1,1],[0,0,1,0],[1,0,0,0],[0,0,0,1],[1,1,0,0],[1,0,1,1],[0,1,1,1],[1,0,1,0],[0,0,0,1],[1,1,0,1],[1,1,1,0],[0,1,1,1],[0,0,1,0],[1,0,0,0],[1,1,0,1],[1,1,1,1],[0,1,1,0],[1,0,0,1],[1,1,1,1],[1,1,0,0],[0,0,0,0],[0,1,0,1],[1,0,0,1],[0,1,1,0],[1,0,1,0],[0,0,1,1],[0,1,0,0],[0,0,0,0],[0,1,0,1],[1,1,1,0],[0,0,1,1]],dtype=np.uint8)
    s5=np.array([[1,1,0,0],[1,0,1,0],[0,0,0,1],[1,1,1,1],[1,0,1,0],[0,1,0,0],[1,1,1,1],[0,0,1,0],[1,0,0,1],[0,1,1,1],[0,0,1,0],[1,1,0,0],[0,1,1,0],[1,0,0,1],[1,0,0,0],[0,1,0,1],[0,0,0,0],[0,1,1,0],[1,1,0,1],[0,0,0,1],[0,0,1,1],[1,1,0,1],[0,1,0,0],[1,1,1,0],[1,1,1,0],[0,0,0,0],[0,1,1,1],[1,0,1,1],[0,1,0,1],[0,0,1,1],[1,0,1,1],[1,0,0,0],[1,0,0,1],[0,1,0,0],[1,1,1,0],[0,0,1,1],[1,1,1,1],[0,0,1,0],[0,1,0,1],[1,1,0,0],[0,0,1,0],[1,0,0,1],[1,0,0,0],[0,1,0,1],[1,1,0,0],[1,1,1,1],[0,0,1,1],[1,0,1,0],[0,1,1,1],[1,0,1,1],[0,0,0,0],[1,1,1,0],[0,1,0,0],[0,0,0,1],[1,0,1,0],[0,1,1,1],[0,0,0,1],[0,1,1,0],[1,1,0,1],[0,0,0,0],[1,0,1,1],[1,0,0,0],[0,1,1,0],[1,1,0,1]],dtype=np.uint8)
    s6=np.array([[0,1,0,0],[1,1,0,1],[1,0,1,1],[0,0,0,0],[0,0,1,0],[1,0,1,1],[1,1,1,0],[0,1,1,1],[1,1,1,1],[0,1,0,0],[0,0,0,0],[1,0,0,1],[1,0,0,0],[0,0,0,1],[1,1,0,1],[1,0,1,0],[0,0,1,1],[1,1,1,0],[1,1,0,0],[0,0,1,1],[1,0,0,1],[0,1,0,1],[0,1,1,1],[1,1,0,0],[0,1,0,1],[0,0,1,0],[1,0,1,0],[1,1,1,1],[0,1,1,0],[1,0,0,0],[0,0,0,1],[0,1,1,0],[0,0,0,1],[0,1,1,0],[0,1,0,0],[1,0,1,1],[1,0,1,1],[1,1,0,1],[1,1,0,1],[1,0,0,0],[1,1,0,0],[0,0,0,1],[0,0,1,1],[0,1,0,0],[0,1,1,1],[1,0,1,0],[1,1,1,0],[0,1,1,1],[1,0,1,0],[1,0,0,1],[1,1,1,1],[0,1,0,1],[0,1,1,0],[0,0,0,0],[1,0,0,0],[1,1,1,1],[0,0,0,0],[1,1,1,0],[0,1,0,1],[0,0,1,0],[1,0,0,1],[0,0,1,1],[0,0,1,0],[1,1,0,0]],dtype=np.uint8)
    s7=np.array([[1,1,0,1],[0,0,0,1],[0,0,1,0],[1,1,1,1],[1,0,0,0],[1,1,0,1],[0,1,0,0],[1,0,0,0],[0,1,1,0],[1,0,1,0],[1,1,1,1],[0,0,1,1],[1,0,1,1],[0,1,1,1],[0,0,0,1],[0,1,0,0],[1,0,1,0],[1,1,0,0],[1,0,0,1],[0,1,0,1],[0,0,1,1],[0,1,1,0],[1,1,1,0],[1,0,1,1],[0,1,0,1],[0,0,0,0],[0,0,0,0],[1,1,1,0],[1,1,0,0],[1,0,0,1],[0,1,1,1],[0,0,1,0],[0,1,1,1],[0,0,1,0],[1,0,1,1],[0,0,0,1],[0,1,0,0],[1,1,1,0],[0,0,0,1],[0,1,1,1],[1,0,0,1],[0,1,0,0],[1,1,0,0],[1,0,1,0],[1,1,1,0],[1,0,0,0],[0,0,1,0],[1,1,0,1],[0,0,0,0],[1,1,1,1],[0,1,1,0],[1,1,0,0],[1,0,1,0],[1,0,0,1],[1,1,0,1],[0,0,0,0],[1,1,1,1],[0,0,1,1],[0,0,1,1],[0,1,0,1],[0,1,0,1],[0,1,1,0],[1,0,0,0],[1,0,1,1]],dtype=np.uint8)
    
    return [s0,s1,s2,s3,s4,s5,s6,s7]

def obtain_subkey_use_L3(c,c1):
    
    s_box=S_box()

    cipher=c
    cipher1=c1

    score_k=[]
    Ch=np.array(cipher[:,:32])
    Cl=np.array(cipher[:,32:])
    
    Ch1=np.array(cipher1[:,:32])
    Cl1=np.array(cipher1[:,32:])
    
    for k in range(2**6):
        Guess_k=num2bitarray6(k)
        Cl_extend=Cl[:,[31,0,1,2,3,4]]#
        Cl1_extend=Cl1[:,[31,0,1,2,3,4]]#
        
        for j in range(len(Cl_extend)):
            Cl_extend[j]=Cl_extend[j]^Guess_k
            Cl1_extend[j]=Cl1_extend[j]^Guess_k
            
        buffer_k=bit2num(Cl_extend)
        buffer_k=s_box[0][buffer_k]
        
        buffer_k1=bit2num(Cl1_extend)
        buffer_k1=s_box[0][buffer_k1]
        
        behind_part=Cl[:,31-7]^Cl[:,31-18]^Cl[:,31-24]^Cl[:,31-29]^Ch[:,31-15]^buffer_k[:,1]^Cl1[:,31-7]^Cl1[:,31-18]^Cl1[:,31-24]^Cl1[:,31-29]^Ch1[:,31-15]^buffer_k1[:,1]
        
        sample=np.array(behind_part,dtype=np.uint32)

        score_k.append(abs(0.5*len(sample)-sum(sample)))

    

    guess_key=np.where(np.array(score_k,dtype=np.float32)==max(np.array(score_k,dtype=np.float32)))[0][0]
    
    guess_key=bin(guess_key)[2:]
    guess_key=guess_key.zfill(6)
    guess_key=list(guess_key)
    guess_key=np.array(guess_key,dtype=np.uint8)
    
    
    
    return guess_key


def attack_4_round_use_L3():
    Sample_pair=100#
    Pc_pair=4#
    num_plaintext=Sample_pair*Pc_pair
    
    #
    keys = np.frombuffer(urandom(64), dtype=np.uint8).reshape(-1,64)
    keys = keys & 1
    subkey=ciph.expand_key(keys,get_Round()+1)#
    
    #
    plain = np.frombuffer(urandom(64*num_plaintext), dtype=np.uint8).reshape(-1,64)
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
    cipher=np.concatenate((Pl,Ph),axis=1)#
    cipher1=np.concatenate((Pl1,Ph1),axis=1)#
    
    real_key=np.zeros(6,dtype=np.uint8)

    real_key[0]=subkey[-1][0,0]
    real_key[1]=subkey[-1][0,1]
    real_key[2]=subkey[-1][0,2]
    real_key[3]=subkey[-1][0,3]
    real_key[4]=subkey[-1][0,4]
    real_key[5]=subkey[-1][0,5]
    
    guess_key=obtain_subkey_use_L3(cipher,cipher1)
    
    
    print('Traditional real key '+str(num_plaintext),real_key,end='----')
    print('guess key',guess_key)
    
    return real_key,guess_key
    
         
num=10**5
flag=0

for i in range(num):   
    real_key,guess_key=attack_4_round_use_L3()

    r=(real_key[0]<<5)+(real_key[1]<<4)+(real_key[2]<<3)+(real_key[3]<<2)+(real_key[4]<<1)+(real_key[5]<<0)
    g=(guess_key[0]<<5)+(guess_key[1]<<4)+(guess_key[2]<<3)+(guess_key[3]<<2)+(guess_key[4]<<1)+(guess_key[5]<<0) 
    if(r==g):
        flag=flag+1
    print(i,flag/(i+1))

print(flag/num)
    
    
    