# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 11:53:09 2019

@author: Adiel
"""
import numpy as np
import scipy.sparse as ssp
from scipy.sparse import linalg
from scipy.sparse import csr_matrix as SM
from scipy.sparse import coo_matrix as CM
#from scipy.stats import unitary_group
import scipy 
from scipy.sparse import dia_matrix
from scipy.sparse import hstack,vstack
import matplotlib.pyplot as plt
import pandas as pd
import time

def make_P(M): 
    n=M.shape[0]
    M1=SM.copy(M)
    M1.data=M.data**2
    M_norms=M1.sum(1)
    M=hstack((np.ones((n,1)),M_norms,-2*M))
    return M

def squaredis(P,Cent):    
    d=Cent.shape[1]
    C=SM((Cent.shape[0],d+2))    
    C[:,1]=1      #C is defined just as in the algorithm you sent me.
    C[:,0] =SM.sum(SM.power(Cent, 2), 1)
    C[:,2:d+2]=Cent
    D=SM.dot(P,C.T)
    D=D.toarray()
    Tags=D.argmin(1)#finding the most close centroid for each point 
    if min(D.shape)>1:
        dists=D.min(1)
    else:
        dists=np.ravel(D)
    y=D.argmin(0)
    return dists,Tags,y 

def make_P_dense(M):
    d=M.shape[1]
    P=np.zeros((M.shape[0],d+2))
    p=np.sum(np.power(M, 2), 1)
    P[:,1:2]=np.reshape(p,(len(p),1)) #P defined just as in the algorithm you sent me
    P[:,0]=1                        
    P[:,2:d+2]=-2*M
    return P    
def squaredis_dense(P,Cent,to_pert=0):
    if len(Cent.shape)==3:
        Cent=np.reshape(Cent,(Cent.shape[0],Cent.shape[2]))    
    d=Cent.shape[1]
    C=np.zeros((Cent.shape[0],d+2))    
    C[:,1]=1      #C is defined just as in the algorithm you sent me.
    cent1=np.copy(Cent)
    print('Cent',type(Cent))
    cent1=np.power(Cent,2)
    c=np.sum(cent1, 1)
    C[:,0:1] =np.reshape(c,(len(c),1))
    C[:,2:d+2]=Cent
    D=np.dot(P,np.transpose(C))
    D[D<0]=0
    if to_pert>0:
        D=D+to_pert*np.random.rand(D.shape[0],D.shape[1])
    Tags=D.argmin(1)  #finding the most close centroid for each point 
    dists=D.min(1)
    y=D.argmin(0)
    y=np.reshape(y,(len(y),1))
    return dists,Tags,y

def kmeans_plspls1(A,w,eps,V,clus_num,we,alfa_app,is_sparse,is_jl):
        """
        This funtion operates the kmeans++ initialization algorithm. each point chosed under the Sinus probability.
        Input:
            A: data matrix, n points, each on a sphere of dimension d.
            k: number of required points to find.
        Output:
            Cents: K initial centroids, each of a dimension d.
        """
        if is_sparse==1:
            A=SM(A)
        if is_jl==1:
            dex=int(clus_num*np.log(A.shape[0]))
    
            ran=np.random.randn(A.shape[1],dex)
            A=SM.dot(A,ran)
            is_sparse=0      #A=np.multiply(w1,A)
        num_of_samples = A.shape[0]
        if any(np.isnan(np.ravel(w)))+any(np.isinf(np.ravel(w))):
            Cents= A[np.random.choice(num_of_samples,size=1),:]   #choosing arbitrary point as the first               
        else: 
            w[w<0]=0               
            Cents= A[np.random.choice(num_of_samples,size=1,p=np.ravel(w)/np.sum(np.ravel(w))),:] #choosing arbitrary point as the first               
        if is_sparse==1:
            PA=make_P(A)
        else:
            PA=make_P_dense(A)
        fcost=alfa_app*1.1
        h1=1
        inds=[]
        while (Cents.shape[0]<clus_num+1):
            Cents2=Cents[h1-1:h1,:] 
            if is_sparse==1:
                Pmina,tags,_=squaredis(PA,Cents2)  
            else:
                Pmina,tags,_=squaredis_dense(PA,Cents2)  
            if h1==1:
                Pmin=Pmina
            else:
                Pmin=np.minimum(Pmin,Pmina)
                Pmin[np.asarray(inds)]=0
            Pmin[Pmin<0]=0
            Pmin00=np.multiply(w,Pmin)
            Pmin0=Pmin00/np.sum(Pmin00)
            if any(np.isnan(np.ravel(Pmin0)))+any(np.isinf(np.ravel(Pmin0))):
                ind=np.random.choice(Pmin.shape[0],1)
            else:
                Pmin0[Pmin0<0]=0
                ind=np.random.choice(Pmin.shape[0],1, p=Pmin0)
            if is_sparse==1:
                Cents=vstack((Cents,A[ind,:]))
            else:
                Cents=np.concatenate((Cents,A[ind,:]),0)
            inds.append(ind)
            h1=h1+1
        return Cents,inds

 
def Lloyd_iteration2( A,P, w ,Q):
    dists,Tags,_=squaredis(P,Q)
    print('finish squaredis')
    Qjl=SM((Q.shape[0],A.shape[1]))
    wq=np.zeros((Q.shape[0],1))
    w=np.reshape(w,(len(w),1))
    for i in range (Qjl.shape[0]):
            #print(i)
            inds=np.where(Tags==i)[0]  

            wmin=0
            wi=w[inds,:]-wmin
            Qjl[i,:]=(A[inds,:].multiply(wi)).sum(0)
            wq[i,:]=np.sum(wi,0)
    wq[wq==0]=1
    wqi=1/wq
    Qjl=Qjl.multiply(wqi+wmin)
    return SM(Qjl)
                        
def k_means_clustering( A,  w ,K, iter_num,exp=1,ind=[],is_sparse=0,is_kline=0,): 

    if ind==[]:    
        ind=np.random.permutation(len(w))[0:K]
    Qnew=A[ind,:]
    P=make_P(A)
    dists1=0
    if (iter_num>=1)+(iter_num==0):
        for i in range(0,iter_num):
            Qnew=Lloyd_iteration2(A,P,w,Qnew) 
            dists0=dists1
            dists1,Tags1,tagss=squaredis(P,Qnew) 
            conv=np.abs(np.sum(np.multiply(w,dists0))-np.sum(np.multiply(w,dists1)))/np.sum(np.multiply(w,dists1))
            print('conv',conv)

    else:     
        Qjl=np.zeros(Qnew.shape)   
        dists0=0
        dists1,Tags1,tagss=squaredis(P,Qnew)    
        i=0        
        conv=np.abs(np.sum(np.multiply(w,dists0))-np.sum(np.multiply(w,dists1)))/np.sum(np.multiply(w,dists1))
        while conv>iter_num:
            Qjl=Qnew
            Qnew=Lloyd_iteration2(A,P,w,Qjl)    
            i=i+1      
            dists0=dists1
            dists1,Tags1,tagss=squaredis(P,Qnew)
            print(np.sum(np.multiply(w,dists1))/500)
            conv=np.abs(np.sum(np.multiply(w,dists0))-np.sum(np.multiply(w,dists1)))/np.sum(np.multiply(w,dists1))
            print('conv',i)
    print('&&&&&&',len(np.unique(tagss)))
    if exp==0:
        Q=SM(A)[tagss,:]
    else:
        Q=Qnew
    return Q,w 

def Coreset_FBL(P,w,B,is_sparse=0):
    w=np.ravel(w)
    #print('wwwwww',w.shape)
    """
    Input:
    P: Data matrix n*d
    w: n weights
    Bsize: size of beta approximation sampling
    m: size of coreset
    alg: algorithm to operate: 0- Benchmark 1-ours 2-ransac
    Output:
    S: coreset matrix m*d
    S_ind: m indeices of rows chosen
    u: sensitivity of every point  
    """
    Bsize=B.shape[0]
    partition=[]

    if is_sparse==1:
        P2=make_P(P)#.multiply(w1)
        dists,Tags,_=squaredis(P2,B)
    else:
        P2=make_P_dense(P)#.multiply(w1)
        print(type(P2))
        print(type(B))

        dists,Tags,t=squaredis_dense(P2,B)
    dists[dists<0]=0
    sum_weights_cluster=np.zeros(Bsize)
    for t in range (0,Bsize):
        sum_weights_cluster[t]=np.sum(w[np.where(Tags==t)[0]])
        partition.append(np.where(Tags==t)[0])

    sumall=2*np.sum(np.multiply(w,dists))
    sumwei=2*Bsize*sum_weights_cluster[Tags]        
    A=np.multiply(w,dists)/sumall
    AA=np.divide(w,sumwei)
    Prob=AA+A
    return Prob,partition,sum_weights_cluster

#def FBL_median(Prob,P,w,Q,B,partition,sum_weights_cluster,coreset_size1,posi,is_sparse):
#    coreset_size=coreset_size1-B.shape[0]
#    S_ind=np.random.choice(len(Prob),coreset_size,p=Prob)
#    uw=np.divide(np.ravel(w),Prob)/coreset_size
#    if posi==1:
#        uw=FBL_positive(P,uw,B,Q,0.1,is_sparse)
#    u0=uw[S_ind]
#    u2=np.zeros(len(partition))
#    for i in range(len(partition)):
#        u2[i]=np.sum(w[np.intersect1d(S_ind,partition[i])])
#    u1=sum_weights_cluster-u2
#    return S_ind,np.concatenate((u0,u1),0)

def FBL_positive(P,u,B,Q,epsit,is_sparse=1):
     if is_sparse==0:    
         BB=make_P_dense(B)
         P0=make_P_dense(P)

         d1,tags,_=squaredis_dense(P0,B)
         d2,_,_=squaredis_dense(BB[tags,:],Q)
     else:
         BB=SM(make_P(B))
         P0=SM(make_P(P))

         d1,tags,_=squaredis(P0,B)
         d2,_,_=squaredis(BB[np.ravel(tags),:],Q)
     d=d1/epsit-d2
     print('d zeroing fraction',len(np.where(d<0)[0])/len(d))
  #   print('d',len(d))
 #    print('u',len(u))
#     print('P',len(P))
     u[np.where(d<0)[0]]=0
     return u
def FBL(P0,P,Prob,partition,sum_weights_cluster,w,indsB,Q,coreset_size,is_not_sparse,full_sampling,posi,eps=0.1):
    Prob=Prob/np.sum(Prob)
    if is_not_sparse==0:
        P0=SM(P0)
    if full_sampling==1:
        ind=np.random.choice(np.arange(len(Prob)),coreset_size,p=np.ravel(Prob)) 
        u=np.divide(np.ravel(w),Prob)/coreset_size    
       #u[np.where(u=='nan')[0]==0]=0
        if posi==1:
            print('is_not_sparse',is_not_sparse)
            if is_not_sparse==0:
                u=FBL_positive(P0,u,P0[np.ravel(indsB),:],Q,eps,1-is_not_sparse)
            else:
                u=FBL_positive(P,u,P[np.ravel(indsB),:],Q,eps,1-is_not_sparse)
        u1=u[ind]
        print('uuuuuu',u[0:10])
        print('uuuuuu1',u1[0:10])

        u1=np.reshape(u1,(u1.shape[0],1))  
        
    else:

            #ind,u1=FBL_median(Prob,P,w,Q,P[np.ravel(indsB),:],partition,sum_weights_cluster,coreset_size,posi,1-is_not_sparse)
        ind=np.random.choice(np.arange(len(Prob)),coreset_size-len(indsB),p=np.ravel(Prob))
        u=np.divide(np.ravel(w),Prob)/coreset_size
        print('ttttuuuuuuttttt',len(u))
        ub=np.zeros(len(indsB))
        if is_not_sparse==1:
            PP0=make_P_dense(P0)
            _,tags,_=squaredis_dense(PP0,P0[indsB,:])
        else:
            PP0=make_P(P0)
            _,tags,_=squaredis(SM(PP0),SM(P0[np.ravel(indsB),:]))
        #print('taggggggggs',tags,len(tags))
        for i in range(len(indsB)):        
            inte=np.intersect1d(ind,np.where(tags==i)[0])
            #ubc[i]=np.sum(w[inte])
            ub[i]=np.sum(w[np.where(tags==i)[0]])-np.sum(u[inte])
            #ub[i]=np.sum(w[indsB[i]])
            #if indsB[i] in inte:
            #    ub[i]=ub[i]-u[indsB[i]]
        #ub=np.abs(ub)
        u1=np.concatenate((u[ind],ub))
        print('ttttuuuuuuttttt1',len(u1))

        if posi==1:
            print('is_not_sparse',is_not_sparse)
            if is_not_sparse==0:
                u1=FBL_positive(vstack((P0[ind,:],P0[np.ravel(indsB),:])),u1,P0[np.ravel(indsB),:],Q,eps,1-is_not_sparse)
            else:
                u1=FBL_positive(np.concatenate((P[ind,:],P[np.ravel(indsB),:]),0),u1,P0[np.ravel(indsB),:],Q,eps,1-is_not_sparse)

    ind=ind.astype(int)
    u1=np.reshape(u1,(len(u1),1))
        
    if full_sampling==0:
        if is_not_sparse==0:
            print('indsBra',np.ravel(indsB))
            print('indsBsh',np.ravel(indsB).shape)

            X=vstack((P0[np.ravel(ind),:],P0[np.ravel(indsB),:]))
        else:
            X=np.concatenate((P0[np.ravel(ind),:],P0[np.ravel(indsB),:]),0)
    else:
            X=P0[ind,:]
    if is_not_sparse==0:
            print(u1.shape)
            print(X.shape)

            C=X.multiply(u1[:X.shape[0],:])
    else:
           C=np.multiply(u1[:X.shape[0]],X)    
    print('Csh',C.shape[0])
    return C,u1[:X.shape[0]],X #for streaming flip X and C.
      
def clus_streaming(path,Data,j,is_pca,alg,h,spar,trial=None,datum=None,is_jl=1,gamma1=0.000000001):
    """
    alg=0 unif sampling
    alg=1 Sohler
    alg=2 CNW
    alg=3 Alaa
    """
    sizeB=j
    coreset_size=Data.shape[0]//(2**(h+1))
    k=0
    T_h= [0] * (h+1) #line 5
    DeltaT_h= [0] * (h+1) #line 4
    u_h=[0]* (h+1) #line 4
    leaf_ind=np.zeros(h+1)
    iter_num=1
    for jj in range(np.power(2,h)): #over all of the leaves
        w=np.ones(2*coreset_size)
        Q0=Data[k:k+2*coreset_size,:]       
        if alg>0: 
            B,inds= kmeans_plspls1(Q0,np.ravel(w),0,[],sizeB,np.ravel(w),0.01,1,0)
            Prob,partition,sum_weights_cluster=Coreset_FBL(Q0,w,B,1)
        if alg>1: 
            Q1,dists11=k_means_clustering(Q0,w,j,iter_num,inds)
        k=k+2*coreset_size
        print('k',k)
        #line 10
        if alg==0: 
            ind=np.random.choice(Q0.shape[0],coreset_size)
            T=Q0[ind,:]
            w=w[0]*np.ones((T.shape[0],1))#*2
        if alg==1:
            _,w,T=FBL(Q0,Q0,Prob,partition,sum_weights_cluster,w,inds,[],coreset_size,0,1,0)
            #w=w*2
        if alg==2:
            _,w,T=FBL(Q0,Q0,Prob,partition,sum_weights_cluster,w,inds,Q1,coreset_size,0,0,1,0.00001)
            #w=np.sqrt(w)
            print('w',w)
        if alg==3:
            _,w,T=FBL(Q0,Q0,Prob,partition,sum_weights_cluster,w,inds,Q1,coreset_size,0,0,1,0.3)
        if alg==4:
            _,w,T=FBL(Q0,Q0,Prob,partition,sum_weights_cluster,w,inds,Q1,coreset_size,0,1,1)
        DeltaT=0
        i=0                        
        u_h[0]=w
        # line 13
        while (i<h)*(type(T_h[i])!=int): #every time the leaf has a neighbor leaf it should merged and reduced
            wT=np.concatenate((w,np.asarray(u_h[i])),0) #line 14
            #line 15 union
            if spar==0:
               totT0=np.concatenate((T,np.asarray(T_h[i])),0)
            else: 
               totT0=vstack((T,T_h[i]))
            totT0=SM(totT0)
            #line 15
            if alg>0:
                B,inds= kmeans_plspls1(totT0,np.ravel(wT),0,[],sizeB,np.ravel(wT),0.01,1,0)
                Prob,partition,sum_weights_cluster=Coreset_FBL(totT0,wT,B,1)  
            if alg>2:
                Q1,dists11=k_means_clustering(totT0,wT,j,iter_num,inds)
            if alg==0:
                T=totT0[np.random.choice(totT0.shape[0],coreset_size),:]
                w=w[0]*np.ones((T.shape[0],1))#*2
            if alg==1:
                T1,w,T=FBL(totT0,totT0,Prob,partition,sum_weights_cluster,wT,inds,[],coreset_size,0,1,0)
                #w=w*2
            if alg==2:
                T1,w,T=FBL(totT0,totT0,Prob,partition,sum_weights_cluster,wT,inds,[],coreset_size,0,0,0)
                #w=np.sqrt(w)
            if alg==3:
                T1,w,T=FBL(totT0,totT0,Prob,partition,sum_weights_cluster,wT,inds,Q1,coreset_size,0,1,1)
            if alg==4:
                T1,w,T=FBL(totT0,totT0,Prob,partition,sum_weights_cluster,wT,inds,Q1,coreset_size,0,0,1)
            DeltaT=0  
            u_h[i]=0
            DeltaT=DeltaT+0 #zeroing leaf which reduced
            T_h[i]=0
            DeltaT_h[i]=0
            leaf_ind[i]=leaf_ind[i]+1
            i=i+1
        T_h[i]=T
        u_h[i]=w        
        T1=T.multiply(w)
        #saving all leaves
        if spar==0:            
            if datum==0:
                np.save(path+'leaves_gyro1/trial='+str(trial)+',j='+str(j)+',alg='+str(alg)+',floor='+str(i)+',leaf='+str(leaf_ind[i])+'.npy',T)
            if datum==1:
                np.save(path+'leaves_acc1/trial='+str(trial)+',j='+str(j)+',alg='+str(alg)+',floor='+str(i)+',leaf='+str(leaf_ind[i])+'.npy',T)
            if datum==2:
                np.save(path+'leaves_mnist/trial='+str(trial)+',j='+str(j)+',alg='+str(alg)+',floor='+str(i)+',leaf='+str(leaf_ind[i])+'.npy',T)
        else:
                ssp.save_npz(path+'trial='+str(trial)+',j='+str(j)+',alg='+str(alg)+',floor='+str(i)+',leaf='+str(leaf_ind[i])+'.npz',T)
                np.save(path+'trial='+str(trial)+',j='+str(j)+',alg='+str(alg)+',floor='+str(i)+',leaf='+str(leaf_ind[i])+'_weights.npy',w)
        DeltaT_h[i]=DeltaT
        Q=[]        
#    if type(T_h[h])==int: #should be remained only the upper one. if not:
    #all_levels=[]
#        for g in range (h+1): #collecting all leaves which remained on tree.
#            if type(T_h[g])!=int:
#                if all_levels==[]:
#                   all_levels=np.asarray(T_h[g])
#                else:
#                    all_levels=np.concatenate((all_levels,np.asarray(T_h[g])),0)
#        DeltaT_hs=sum(DeltaT_h[h]) #summing its delta
#    else:
#        all_levels=T_h[h] 
#        DeltaT_hs=DeltaT_h[h]
    return []

def old_clustering1( A,w,alfa_app,eps,V, K,is_sparse,is_plspls=0,is_klinemeans=0):
        
        """
     
        inputs:
            A: data matrix, n points, each of dimension d.
            K: number of centroids demanded for the Kmeans.
            is_sparse: the  output SA0 will be: '0' the accurate cantroids, '1' the points that are the most close to the centroids.
            is_plspls: '1' to initialize with the kmeans++ algorithm which bounds the error, '0' random initialization.
            is_klinemeans:  '1' calculates klinemeans, '0' calculates Lloyd's kmeans.
        
        output:
            SA0: "ready coreset": a matrix of size K*d: coreset points multiplies by weights.
            GW1: weights
            Tags1: Data indices of the points chosen to coreset.
    
    """ 
        #sensitivity=0.01
        num_of_samples = A.shape[0]
        
        if is_klinemeans==1:
            if is_sparse==0:
                A1,weights1=nor_data(A)
            else:
                A1,weights1=nor_data1(A)
            weights1=np.reshape(weights1,(len(weights1),1))
            weights=np.multiply(w,weights1)
        else:
            if is_sparse==0:
                A1=np.copy(A)
            else:
                A1=SM.copy(A)
            weights=w
        print('A1',type(A1))
        print('A1',type(A1.shape[0]))
        print('A1',type(A1.shape[1]))

        num_of_samples = A1.shape[0]
        num_of_channels = A1.shape[1]
        K=int(K)
        if is_sparse==0:
            P=make_P_dense(A1)       
            Cent=np.zeros((2*K,num_of_channels))
        else:
            P=make_P(A1)       
            Centt=SM((2*K,num_of_channels))
        if is_plspls==1:
            Centt,per=kmeans_plspls1(A1,np.ravel(np.power(weights,2)),eps,V,K,np.power(weights,2),alfa_app,is_sparse,is_jl=0)            
        else:
            per=np.random.permutation(num_of_samples)
            #Cent[0:K,:]=A1[per[0:K],:]
        if is_sparse==0:
            #Cent=A1[np.ravel(per[0:K]),:]
            print('****per****',len(np.unique(per)))
            Cent=np.concatenate((A1[np.ravel(per[0:K]),:],A1[np.ravel(per[0:K]),:]),0)
        else:
            Cent=vstack((A1[np.ravel(per[0:K]),:],A1[np.ravel(per[0:K]),:]))
            #Cent[0:K,:]=A1[np.ravel(per[0:K]),:]
            print('****per****',len(np.unique(per)))
        K1=Cent.shape[0]
    
        
        iter=0
        Cost=50 #should be just !=0
        old_Cost=2*Cost
    
        Tags=np.zeros((num_of_samples,1)) # a vector stores the cluster of each point
        print('c0s',Cent.shape)
        sensitivity=0.01
        it=0
        while np.logical_or(it<1,np.logical_and(min(Cost/old_Cost,old_Cost/Cost)<sensitivity,Cost>0.000001)): #the corrent cost indeed resuces relating the previous one, 
        #for i in range(10):
                            #however the loop continues until the reduction is not significantly and their ratio is close to one, and exceeds the parameter "sensitivity"    
            group_weights=np.zeros((K1,1))
            iter=iter+1 #counting the iterations. only for control
            old_Cost=Cost #the last calculated Cost becomes the old_Cost, and a new Cost is going to be calculated.
            if is_sparse==0:            
                Cent1=np.copy(Cent)
                Dmin,Tags,Tags1=squaredis_dense(P,Cent1)
            else:
                Cent1=SM.copy(Cent)
                Dmin,Tags,Tags1=squaredis(P,Cent1)
            #print('Tags',Tags)
            Cost=np.sum(Dmin) #the cost is the summation of all of the minimal distances
            for kk in range (1,K1+1):
                wheres=np.where(Tags==kk-1)  #finding the indeces of cluster k
                #print('wheres',weights[wheres[0]])
                weights2=np.power(weights[wheres[0]],1)  #finding the weights of cluster k
                group_weights[kk-1,:]=np.sum(weights2)
              
            it=it+1           
            
        GW1=np.power(group_weights,1)
        print('***GW1***',len(np.where(GW1>0)[0]))
        F=Cent
        if is_sparse==0:
            
            SA0=np.multiply(GW1,F) #We may weight each group with its overall weight in ordet to compare it to the original data.   
        else:
            SA0=F.multiply(GW1)
        print('SA0',SA0)
        return SA0,GW1,Tags1    
    
def old_clustering( A,w,alfa_app,eps,V, K,is_sparse,is_plspls=0,is_klinemeans=0):
        
        """
     
        inputs:
            A: data matrix, n points, each of dimension d.
            K: number of centroids demanded for the Kmeans.
            is_sparse: the  output SA0 will be: '0' the accurate cantroids, '1' the points that are the most close to the centroids.
            is_plspls: '1' to initialize with the kmeans++ algorithm which bounds the error, '0' random initialization.
            is_klinemeans:  '1' calculates klinemeans, '0' calculates Lloyd's kmeans.
        
        output:
            SA0: "ready coreset": a matrix of size K*d: coreset points multiplies by weights.
            GW1: weights
            Tags1: Data indices of the points chosen to coreset.
    
    """ 
        #sensitivity=0.01
        num_of_samples = A.shape[0]
        
        if is_klinemeans==1:
            if is_sparse==0:
                A1,weights1=nor_data(A)
            else:
                A1,weights1=nor_data1(A)
            weights1=np.reshape(weights1,(len(weights1),1))
            weights=np.multiply(w,weights1)
        else:
            if is_sparse==0:
                A1=np.copy(A)
            else:
                A1=SM.copy(A)
            weights=w
        print('A1',type(A1))
        print('A1',type(A1.shape[0]))
        print('A1',type(A1.shape[1]))

        num_of_samples = A1.shape[0]
        num_of_channels = A1.shape[1]
        K=int(K)
        if is_sparse==0:
            P=make_P_dense(A1)       
            Cent=np.zeros((K,num_of_channels))
        else:
            P=make_P(A1)       
            Centt=SM((K,num_of_channels))
        if is_plspls==1:
            Centt,per=kmeans_plspls1(A1,np.ravel(np.power(weights,2)),eps,V,K,np.power(weights,2),alfa_app,is_sparse,is_jl=0)            
        else:
            per=np.random.permutation(num_of_samples)
            #Cent[0:K,:]=A1[per[0:K],:]
        if is_sparse==0:
            #Cent=A1[np.ravel(per[0:K]),:]
            print('****per****',len(np.unique(per)))
            Cent=np.concatenate((A1[np.ravel(per[0:K]),:],A1[np.ravel(per[0:K]),:]),0)
        else:
            #Cent=vstack((A1[np.ravel(per[0:K]),:],A1[np.ravel(per[0:K]),:]))
            Cent=A1[np.ravel(per[0:K]),:]
            print('****per****',len(np.unique(per)))
        K1=Cent.shape[0]
    
        
        iter=0
        Cost=50 #should be just !=0
        old_Cost=2*Cost
    
        Tags=np.zeros((num_of_samples,1)) # a vector stores the cluster of each point
        print('c0s',Cent.shape)
        sensitivity=0.01
        it=0
        while np.logical_or(it<1,np.logical_and(min(Cost/old_Cost,old_Cost/Cost)<sensitivity,Cost>0.000001)): #the corrent cost indeed resuces relating the previous one, 
        #for i in range(10):
                            #however the loop continues until the reduction is not significantly and their ratio is close to one, and exceeds the parameter "sensitivity"    
            group_weights=np.zeros((K1,1))
            iter=iter+1 #counting the iterations. only for control
            old_Cost=Cost #the last calculated Cost becomes the old_Cost, and a new Cost is going to be calculated.
            if is_sparse==0:            
                Cent1=np.copy(Cent)
                Dmin,Tags,Tags1=squaredis_dense(P,Cent1)
            else:
                Cent1=SM.copy(Cent)
                Dmin,Tags,Tags1=squaredis(P,Cent1)
            #print('Tags',Tags)
            Cost=np.sum(Dmin) #the cost is the summation of all of the minimal distances
            for kk in range (1,K1+1):
                wheres=np.where(Tags==kk-1)  #finding the indeces of cluster k
                #print('wheres',weights[wheres[0]])
                weights2=np.power(weights[wheres[0]],1)  #finding the weights of cluster k
                group_weights[kk-1,:]=np.sum(weights2)
              
            it=it+1           
            
        GW1=np.power(group_weights,1)
        GW1=np.power(group_weights,1)

        print('***GW1***',len(np.where(GW1>0)[0]))
        F=Cent
        if is_sparse==0:
            
            SA0=np.multiply(GW1,F) #We may weight each group with its overall weight in ordet to compare it to the original data.   
        else:
            SA0=F.multiply(GW1)
#        print('SA0',SA0)
        return Cent,[],[]    