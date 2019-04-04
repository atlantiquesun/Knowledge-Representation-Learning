#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 09:36:04 2019

@author: yiransun
"""

import numpy as np
import pandas as pd
from scipy.linalg import *
import xlrd
import tensorflow as tf
from random import shuffle
from sklearn.decomposition import PCA


def check_dd(X):
    '''
    X: numpy.array
    return a boolean value (1: X is diagonally dominant, 0:...)
    '''
    D=np.diag(np.abs(X))
    S=np.sum(np.abs(X),axis=1)-D
    if np.all(D>S):
        return 1
    else:
        return 0

def learning_rate(emb,n,order=-2):
    '''
    emb: a numpy matrix, has n none zero values
    order: the final order we want the 
    '''   
    count=round(abs(np.log(emb)/np.log(10)))+order
    return count

def find_e(s):
    '''
    return the position of 'e' counting from backwards
    '''
    if 'e' not in s: return len(s)
    else:
        return s.find('e')
    

def get_sf_vec(v,n,d=0.1):
    '''
    v: the vector
    n: the dimension of the vector
    d: learning rate
    '''
    v1=v.tolist()
    s=str(v1)
    s=s[1:-1].split(',')
    s1=[x[:find_e(x)] for x in s]
    s2=[]
    count=0
    for i in s:
        ending=''
        mark=len(i)
        for j in range(len(i)):
            if i[j]=='e':
                count+=1
                mark=j
            if j>mark+1:
                ending+=i[j]
        s2.append(ending)
    if count>=n:    
        s2=[x[1:] for x in s2]
        s1=[float(x) for x in s1]
        transformed=[]
        for j in s2:
            if j[0]=='0':
                j=j[1:]
        s2=[int(x) for x in s2]
        for i in range(len(s1)):
            transformed.append(d*s1[i]*(10**(-(s2[i]-min(s2)))))
        return np.asarray(transformed)
    else:
        return v

def get_sf_mat(m,n,d=0.01):
    '''
    m: the matrix
    '''
    v=np.diag(m)
    v=get_sf_vec(v,n,d)
    return np.diag(v)

def get_sf_num(num,d=0.1):
    '''
    m: the number
    '''
    num=str(num)[:find_e(str(num))]
    num=float(num)
    return num
        

class model():
    def __init__(self,n_entities=23634,n_concats=3617,n=20,space_size=30,n_samples=32000,init='aspre',sig_init=7):
        '''
        n_entities: number of entities in total
        n_concats: numbe`r of concats in total
        datasets: (train,test)
        n: dimension of the embedding
        space_size: how scattered we would like the embeddings to be 
        '''
        if init=='asemb':
            train=pd.read_csv('/Users/yiransun/Desktop/data/data_csv/complete/train_complete.csv')
            train=train[train.columns.values[1:]]
            test=pd.read_csv('/Users/yiransun/Desktop/data/data_csv/complete/test_complete.csv')
            test=test[test.columns.values[1:]]
            train=np.asarray(train).astype(int)
            test=np.asarray(test).astype(int)
            train=train.tolist()
            test=test.tolist()
            n_entities=32634
            n_concats=3617
            raw_ent=np.zeros((32634,3617))
            raw_cct=np.zeros((3617,32634))
            for i in train:
                if i[2]==1:
                    raw_ent[i[0]][i[1]]=1
                    raw_cct[i[1]][i[0]]=1
            ent_emb=PCA(n_components=n).fit_transform(raw_ent)
            cct_emb=PCA(n_components=n).fit_transform(raw_cct)
            cct_sig=cct_emb.tolist()
            cct_sig=[np.diag(np.asarray(x)) for x in cct_sig]
            cct_sig=[x.reshape((n,n)) for x in cct_sig]    
            freq=pd.read_csv('/Users/yiransun/Desktop/data/data_csv/complete/concat_2_id_complete.csv')
            indices=list(freq['id'])
            freq=np.asarray(freq['frequency'])/np.max(np.asarray(freq['frequency']))
            mapping=dict(zip(indices,freq))
            for i in range(len(cct_sig)):
                cct_sig[i]=np.abs(mapping[i]*cct_sig[i])
        elif init=='aspre':
            train=pd.read_csv('/Users/yiransun/Desktop/data/data_csv/complete/train_complete.csv')
            train=train[train.columns.values[1:]]
            test=pd.read_csv('/Users/yiransun/Desktop/data/data_csv/complete/test_complete.csv')
            test=test[test.columns.values[1:]]
            train=np.asarray(train).astype(int)
            test=np.asarray(test).astype(int)
            train=train.tolist()
            test=test.tolist()
            
            raw_ent=np.zeros((n_entities,n_concats))
            raw_cct=np.zeros((n_concats,n_entities))
            
            for i in train:
                if i[2]==1:
                    raw_ent[i[0]][i[1]]=1
                    raw_cct[i[1]][i[0]]=1
            
            ent_emb=np.zeros((n_entities,n))
            #ent_emb=PCA(n_components=30).fit_transform(raw_ent)
            cct_emb=PCA(n_components=n).fit_transform(raw_cct)
            for i in range(n_entities):
                mul=np.repeat(raw_ent[i].reshape((n_concats,1)),n,axis=1)
                s=np.multiply(mul,cct_emb)
                k=np.sum(s,axis=0)
                if np.sum(raw_ent[i])!=0:
                    k=k/np.sum(raw_ent[i])
                ent_emb[i]=k
            cct_sig=[n]*n_concats
            cct_sig=[np.diag(np.zeros(x)+0.3) for x in cct_sig]
                
        self.train=train
        self.test=test    
        cct_vol=[1]*n_concats     
        self.ent_emb=ent_emb
        self.cct_emb=cct_emb
        #self.ent_sig=ent_sig
        self.cct_sig=cct_sig
        self.cct_vol=cct_vol
        self.n=n
        self.flag=False
        self.n_samples=n_samples
        self.appearances=np.zeros((n_concats,))+2
    
    def get_datasets(self,which='full'):
        if which=='high_freq':
            train=pd.read_csv('/Users/yiransun/Desktop/data/data_csv/fulldatasets.csv',sep=';')
            train=np.asarray(train)
            (n_train,_)=train.shape
            train=train.tolist()
            self.train=train
        elif which=='full':
            train=pd.read_csv('/Users/yiransun/Desktop/data/data_csv/complete/train_complete.csv')
            train=train[train.columns.values[1:]]
            test=pd.read_csv('/Users/yiransun/Desktop/data/data_csv/complete/test_complete.csv')
            test=test[test.columns.values[1:]]
            train=np.asarray(train).astype(int)
            test=np.asarray(test).astype(int)
            train=train.tolist()
            test=test.tolist()
            self.train=train
            self.test=test
        else:
            return False
    def relative_prob(self,e1,e2):
        '''
        return p(e1|e2)/max p(x|e2)
        '''
        sig2=self.cct_sig[e2]
        e1=self.ent_emb[e1]
        e2=self.cct_emb[e2]
        a=e1-e2
        exp=-0.5*np.dot(np.dot(a,inv(sig2)),a.T)
        prob=(np.e**exp)
        return prob
        
    def cost_function(self,e1,e2):
        '''
        e1: index of the 'element(entity)', int
        e2: index of the 'set(concat)', int
        n: the dimension of e1,e2 and sig2, int
        cost:-p(e1|e2) - needs to be minimized
        pos: 1 if the pair is a correct inclusion pair, 0 if it is not
        '''        
        sig2=self.cct_sig[e2]
        #vol=self.cct_vol[e2]
        e1=self.ent_emb[e1]
        e2=self.cct_emb[e2]  
        a=e1-e2
        exp=-0.5*np.dot(np.dot(a,inv(sig2)),a.T)
        cost=np.e**exp
        return cost
    
    def activation(self,cost,mode='logistics'):
        if mode=='logistics':
            return 1/(1+np.e**(-cost))
        elif (mode=='relu'):
            if cost>0:
                return cost
            else:
                return 0
    
    def loss_function(self,e1,e2,pos):
        cost=self.cost_function(e1,e2)
        activation=self.activation(cost)
        if pos==1:
            loss=-pos*np.log(activation)
        else:
            loss=-(1-pos)*np.log(1-activation)
     
        return (cost,activation,loss)
    
    def compute_gradient(self,e1,e2,pos,cost,activation,loss):
        '''
        loss:loss_function(e1,e2,pos)
        '''
        sig2=self.cct_sig[e2]
        e1=self.ent_emb[e1]
        e2=self.cct_emb[e2]
        a=e1-e2
        Hyhat_grad=(activation-pos)/(activation*(1-activation))
        if (activation*(1-activation))<0.001: #prevent the explosion of gradients
            Hyhat_grad=(activation-pos)/1
        
        if pos==1 and cost==0:
            cost=loss #to resurrect those samples that are "stuck" 
        
        yhatc_grad=(np.e**(-cost))/((1+np.e**(-cost))**2)
       
        cmiuA_grad=-np.dot(inv(sig2),a.T)*cost
    
        cmiuB_grad=np.dot(inv(sig2),a.T)*cost
        
        csigB_grad=0.5*np.dot(np.dot(np.dot((inv(sig2).T),a.T),a),inv(sig2).T)*cost
        
        miuA_grad=Hyhat_grad*yhatc_grad*cmiuA_grad
        miuB_grad=Hyhat_grad*yhatc_grad*cmiuB_grad
        sigB_grad=Hyhat_grad*yhatc_grad*csigB_grad
        
        return (miuA_grad,miuB_grad,sigB_grad)
    
    def parameter_update(self,embs,grads,cost,loss,pos,d=0.01):
        '''
        embs=(e1,e2)
        grads=(e1_grad,e2_grad,sig2_grad)
        '''
        (e1,e2)=embs
        if loss>0.5 and pos==1:
            d=d*5 #to increase the speed of update, resurrect the "dead" samples
        if loss>0.5 and pos==0:
            d=d*5
        if pos==0: d=d*5
        #print(grads)
        if self.appearances[e2]>100:self.appearances[e2]=100
        self.ent_emb[e1] =self.ent_emb[e1]-(get_sf_vec(grads[0],self.n,d=d))
        self.cct_emb[e2] =self.cct_emb[e2]-(get_sf_vec(grads[1],self.n,d=d))
        self.cct_sig[e2] =self.cct_sig[e2]-(1/np.log(self.appearances[e2]))*get_sf_mat(grads[2],self.n,d=d)
    
    def train_model(self,n_batches=40,epoch=30,learning_rate=1,d=3,which='full'):
        '''
        train: [[ent_index,concat_index,pos],...]
        '''
        self.n_batches=n_batches
        self.get_datasets(which=which)
        for r in range(epoch): 
            print("epoch "+str(r+1)+':',end=' ')
            for u in range(n_batches):
                train=self.train[self.n_samples*u:self.n_samples*(u+1)]
                count=0
                ave_loss=0
                for i in train:
                    if i[2]==1:
                        self.appearances[i[1]]+=1
                    (cost,activation,loss)=self.loss_function(i[0],i[1],i[2])
                    count+=1
                    ave_loss=(ave_loss+loss)/count
                    grads=self.compute_gradient(i[0],i[1],i[2],cost,activation,loss)
                    self.parameter_update([i[0],i[1]],grads,cost,loss,i[2],d=d)
                print(u,r)
        
        return "Finished"       
        
    def test_model2(self):        
        correct=0
        for i in self.train[:self.n_samples]:
            prob=self.cost_function(i[0],i[1])
            if prob>(10**(-100)):
                prob=1
            #activation=self.activation(prob)
            if round(prob)==i[2]:
                correct+=1
        print("accuracy: ", correct/self.n_samples)
        
    def test_model3(self,n_batches=2,freq=0,size=2):
        ent_dic={}
        for i in self.train[:n_batches*self.n_samples]:
            if i[0] not in ent_dic:
                ent_dic[i[0]]=1
        ent=list(ent_dic.keys())
                
        test=[]        
        for i in self.train[self.n_batches*self.n_samples:size*self.n_batches*self.n_samples]:
            if i[0] in ent:
                test.append(i)
        
        correct=0        
        for i in test:
            prob=self.cost_function(i[0],i[1])
            if prob>(10**(-100)):
                prob=1
            #activation=self.activation(prob)
            if round(prob)==i[2]:
                correct+=1
        print("accuracy: ", correct/len(test))
    def test_model4(self,freq=0,size=2):
        ent_dic={}
        for i in self.train[:self.n_batches*self.n_samples]:
            if i[0] not in ent_dic:
                ent_dic[i[0]]=1
            else:
                ent_dic[i[0]]+=1
                
        ent=list(ent_dic.keys())
                
        test=[]        
        for i in self.test:
            if i[0] in ent:
                test.append(i)
        
        correct=0        
        for i in test:
            prob=self.cost_function(i[0],i[1])
            if np.log(prob)/np.log(10)>-100:
                prob=1
            #activation=self.activation(prob)
            if round(prob)==i[2]:
                correct+=1
        print("accuracy: ", correct/len(test))
        
#17958-503, 1934-509
#l=[503,498,499,508,509,510, 513,662,707,708,715,717,756,758]
            
m=model(n_samples=500,space_size=1,sig_init=1,n=20,init='aspre')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
