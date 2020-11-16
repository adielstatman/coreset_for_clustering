

"""
Created on Thu Jan  3 10:17:23 2019

@author: Adiel
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 12:37:14 2018

@author: Adiel
147 270 413 567 711

This script generates the experiments of the paper..
"""      

      

def plot_figures(h,xlabel,fig_name,error,ii,is_log=1,loc1=0):
    plt.figure(h)
    #if is_log==1:
    #t.yscale('log')
    plt.title(fig_name)
    lab=['Uniform sampling','Algorithm 1','Algorithm 2','Algorithm 3','Algorithm 2 non-negative']
    #for t in range(0,3):
    plt.plot(error[0,:],np.mean(error[num_of_r*0+1:num_of_r*(0+1)+1,:],0),marker='^',label=lab[0])   
    plt.plot(error[0,:],np.mean(error[num_of_r*1+1:num_of_r*(1+1)+1,:],0),marker='^',label=lab[1])   
    plt.plot(error[0,:],np.mean(error[num_of_r*2+1:num_of_r*(2+1)+1,:],0),marker='^',label=lab[2])   
    plt.plot(error[0,:],np.mean(error[num_of_r*3+1:num_of_r*(3+1)+1,:],0),marker='^',label=lab[3])   

    plt.legend(loc=0)
    plt.xlabel(xlabel)
    plt.xlim((error[0,0],error[0,-1]))
    plt.ylabel("Approximation Error")
    plt.grid() 
from scipy import io
import general_SVD_algs1 as gsc 
import scipy.sparse as ssp
from scipy.sparse import linalg
from scipy.sparse import coo_matrix as CM
from scipy.sparse import csr_matrix as SM
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
datum=1
is_pca=1

Data=np.random.rand(1000,101)
Data=Data/np.max(np.max(Data))
is_sparse=0
is_partial=0
num_of_cent_amount=1#don't touch, need to get to 480
exp_num=7 #want to have 4 coreset sizes for the stds.
n=Data.shape[0]
w=np.ones(n)

iter_num=0.01

d=Data.shape[1]
P2=gsc.make_P_dense(Data)    
num_of_lines=4
num_of_r=3
print('Hi')
k_freq=50#5 is noisy
beg0=time.time()
B0,inds= gsc.kmeans_plspls1(SM(Data),w,0,[],k_freq*num_of_cent_amount,np.ravel(w),0.01,1,0)
inds=np.ravel(np.asarray(inds))
B0=B0.toarray()
print('prod B',type(B0))
print('prod B',time.time()-beg0)
exact=1
error1=np.zeros((num_of_lines+1,num_of_cent_amount))    
error2=np.zeros((num_of_lines+1,num_of_cent_amount))    
error3=np.zeros((num_of_lines+1,num_of_cent_amount))    
error4=np.zeros((num_of_lines+1,num_of_cent_amount))   
 
num_of_clus=np.zeros(num_of_cent_amount)
error=np.zeros((num_of_r*num_of_lines+1,exp_num))  
erroro=np.zeros((num_of_r*num_of_lines+1,exp_num))    
  
coreset_size=np.zeros(exp_num,dtype=int)
for ii in range (0,num_of_cent_amount): 
    sizeB=k_freq*(ii+1)
    k=k_freq*(ii+1)  
    B=B0[0:10,:]
    num_of_clus[ii]=k
    print('prod BB',type(B))

    Prob,partition,sum_weights_cluster=gsc.Coreset_FBL(Data,w,B)  
    beg0=time.time()  
    Q,dists11=gsc.k_means_clustering(SM(Data),w,k,iter_num,exact,inds[0:k])
    Q=Q.toarray()
    dists1,ta,t=gsc.squaredis_dense(P2,Q)
    sum1=np.sum(np.multiply(w,dists1))

    for r in range (1,num_of_r+1):
        print('r=',r)
        for alg in range (0,num_of_lines):        
            
                for m0 in range (1,exp_num+1):         
                    begin0=time.time()
                    m=0.7*Data.shape[0]*m0/exp_num-sizeB

                    coreset_size[m0-1]=int(m+sizeB)
                    if alg==0:
                        S=Data[np.random.randint(Data.shape[0],size=coreset_size[m0-1]),:]
                        u1=np.ones(coreset_size[m0-1])*(len(Data)/coreset_size[m0-1])
                    if alg==1:
                        S1,u1,S=gsc.FBL(Data,Data,Prob,partition,sum_weights_cluster,w,inds[0:k],Q,coreset_size[m0-1],1,1,0)
                    if alg==2:
                        S1,u1,S=gsc.FBL(Data,Data,Prob,partition,sum_weights_cluster,w,inds[0:k],Q,coreset_size[m0-1],1,0,1,0.00001)
                    if alg==3:
                        S1,u1,S=gsc.FBL(Data,Data,Prob,partition,sum_weights_cluster,w,inds[0:k],Q,coreset_size[m0-1],1,0,1,0.3)
                    if alg==4:
                        S1,u1,S=gsc.FBL(Data,Data,Prob,partition,sum_weights_cluster,w,inds[0:k],Q,coreset_size[m0-1],1,1,1)
                    print('now clustering coreset')
                    print('alg',alg)
                    a=np.ravel(u1)                    
                    #a=np.ones((len(u1),1))
                    a[np.where(a<0)[0]]=0
                    a=np.sign(a)
                    B0,indsS= gsc.kmeans_plspls1(SM(S),np.ravel(a),0,[],k,np.ravel(a),0.01,1,0)
                    indsS=np.ravel(np.asarray(indsS))
                    u1=np.reshape(u1,(len(u1),1))
                    Q1,dists3=gsc.k_means_clustering( SM(S), u1 ,k, iter_num, exact, indsS)
                    Q1=Q1.toarray()
                    dists2,ta, t=gsc.squaredis_dense(P2,Q1)
                    u=np.divide(w,Prob)/coreset_size[m0-1]
                    u=u/np.sum(u)
                    sum2=np.sum(np.multiply(w,dists2))
                    print('sum1',sum1)
                    print('sum2',sum2)
                    if ii==0:
                        error[0,:]=coreset_size

                        error[r+alg*num_of_r,m0-1]=(sum2-sum1)/sum1
                    else:
                        erroro[0,:]=coreset_size

                        erroro[r+alg*num_of_r,m0-1]=(sum2-sum1)/sum1



        if r==1:
            print('exp 1',time.time()-beg0)          

fig_name=' k='+str(k_freq)+' B='+str(k_freq)
plot_figures(1,"# sampled points",fig_name,np.abs(error),ii,0,1)

np.save('C:/Users/Adiel/Dropbox/All_experimental/FBL/'+fig_name+'.npy',error)
fig_name=' k='+str(k_freq*2)+' B='+str(k_freq*2)
plot_figures(2,"# sampled points",fig_name,np.abs(erroro),ii,0,1)

np.save('C:/Users/Adiel/Dropbox/All_experimental/FBL/'+fig_name+'.npy',erroro)
