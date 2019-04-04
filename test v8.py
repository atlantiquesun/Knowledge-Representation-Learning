#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 20:23:29 2019

@author: yiransun
"""

import numpy as np
import pandas as pd
from scipy.linalg import *
from random import shuffle, randint
from sklearn.decomposition import PCA

def relu(a):
    k=a.shape[0]
    a=a.reshape((1,k))
    z=np.zeros((1,k))
    return np.max(np.concatenate((a,z),axis=0),axis=0)

def inv_relu(a):
    k=a.shape[0]
    a=a.reshape((1,k))
    z=np.zeros((1,k))
    return np.min(np.concatenate((a,z),axis=0),axis=0)

class model():
    def __init__(self,n_entities=23634,n_concats=3617,n=20,sig_init=0.3):
        '''
        n_entities: number of entities in total
        n_concats: numbe`r of concats in total
        datasets: (train,test)
        n: dimension of the embedding
        space_size: how scattered we would like the embeddings to be 
        '''   
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
        cct_emb=PCA(n_components=n).fit_transform(raw_cct)
        for i in range(n_entities):
            mul=np.repeat(raw_ent[i].reshape((n_concats,1)),n,axis=1)
            s=np.multiply(mul,cct_emb)
            k=np.sum(s,axis=0)
            if np.sum(raw_ent[i])!=0:
                k=k/np.sum(raw_ent[i])
            ent_emb[i]=k
        cct_sig=[n]*n_concats
        cct_sig=[np.diag(np.zeros(x)+sig_init) for x in cct_sig]
        
        
        self.train=train
        self.test=test
        self.ent_emb=ent_emb
        self.cct_emb=cct_emb
        self.cct_sig=cct_sig
        self.n=n
        self.n_concats=n_concats
        self.n_entities=n_entities
        self.trained=[[]]*n_entities
    
    def get_avg_gradient(self,e1,e2,fin_po,sample_size=20):
        gradient=np.zeros((self.n,))
        trained=self.trained[e1]
        if(len(trained)>0):
            for j in range(sample_size):
                i=trained[randint(0,len(trained)-1)]
                c=m.cost_function(e1,i)
                c2=m.cost_external(fin_po,i)
                if c<10**(-5): #hyperparameter warning
                    d1=np.abs(self.cct_emb[i]-self.ent_emb[e1])
                    if c2>c:
                        gradient+=(-1)*c*d1/(np.diag(self.cct_sig[i]))
                    else:
                        gradient+=(1)*c*d1/(np.diag(self.cct_sig[i]))
            d=np.abs(self.cct_emb[e2]-self.ent_emb[e1])
            c=m.cost_function(e1,e2)
            gradient+=(-1)*d*c/(np.diag(self.cct_sig[e2]))
            gradient=relu(gradient)
            #if np.max(np.abs(gradient))>1:
            if np.sum(gradient)!=0:
                gradient=(gradient/np.max(np.abs(gradient)))*0.1
        return gradient                        
    
    def update(self,e1,e2,pos,learning_rate):
        dg=np.sqrt(np.diagonal(self.cct_sig[e2]))
        loss=np.abs(self.cct_emb[e2]-self.ent_emb[e1])-np.sqrt(2)*dg
    
        if pos==1:
            loss=relu(loss)
            if np.sum(loss)!=0:
                loss=loss/np.sum(loss)
            dif=self.cct_emb[e2]-self.ent_emb[e1]
            
            fin_po=self.ent_emb[e1]+0.5*loss*learning_rate*dif
            gradient=self.get_avg_gradient(e1,e2,fin_po)
            
            self.ent_emb[e1]=self.ent_emb[e1]+0.5*loss*(1-gradient)*learning_rate*dif            
            if np.isnan(self.ent_emb[e1]).any(): return 0
            self.cct_emb[e2]=self.cct_emb[e2]-0.5*loss*learning_rate*dif
            self.cct_sig[e2]=np.diag(np.square(dg+loss*learning_rate))
            
        else:
            loss=np.abs(inv_relu(loss))
            if np.sum(loss)!=0:
                loss=loss/np.sum(loss)
            dif=self.cct_emb[e2]-self.ent_emb[e1]
            self.ent_emb[e1]=self.ent_emb[e1]-0.5*loss*learning_rate*dif
            if np.isnan(self.ent_emb[e1]).any(): return 0
            self.cct_emb[e2]=self.cct_emb[e2]+0.5*loss*learning_rate*dif
            self.cct_sig[e2]=np.diag(np.square(dg-loss*learning_rate))
        return 1
    
    def train_model(self,learning_rate=0.1,epoch=10,n_batches=1,batch_size=10,start=0):
        self.n_batches=n_batches
        self.batch_size=batch_size
        for r in range(epoch): 
            print("epoch "+str(r+1)+':',end=' ')
            for u in range(start,start+n_batches):
                train=self.train[self.batch_size*u:self.batch_size*(u+1)]
                for i in train:
                    if i[2]==1:
                        self.trained[i[0]].append(i[1])
                    q=self.update(i[0],i[1],i[2],learning_rate)
                    if q==0: return 0
                print(u,r)        
        return "Finished"
    
    def cost_function(self,e1,e2):
        sig2=self.cct_sig[e2]
        e1=self.ent_emb[e1]
        e2=self.cct_emb[e2]  
        a=e1-e2
        exp=-0.5*np.dot(np.dot(a,inv(sig2)),a.T)
        cost=np.e**exp
        return cost
    
    def cost_external(self,v1,e2):
        sig2=self.cct_sig[e2]
        e2=self.cct_emb[e2]
        e1=v1
        a=e1-e2
        exp=-0.5*np.dot(np.dot(a,inv(sig2)),a.T)
        cost=np.e**exp
        return cost
    
    def test_model(self,size=2):
        ent_dic={}
        for i in self.train[:self.n_batches*self.batch_size]:
            if i[0] not in ent_dic:
                ent_dic[i[0]]=1
            else:
                ent_dic[i[0]]+=1
                
        ent=list(ent_dic.keys())
                
        test=[]        
        for i in self.test[:size*self.n_batches*self.batch_size]:
            if i[0] in ent:
                test.append(i)
        
        correct=0
        scores=[]       
        for i in test:
            prob=self.cost_function(i[0],i[1])
            scores.append([prob,i[2]])
            if np.log(prob)/np.log(10)>-100:
                prob=1
            if round(prob)==i[2]:
                correct+=1
        print("accuracy: ", correct/len(test))
        return scores

  
m=model(n=25)
    
        