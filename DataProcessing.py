import os
import numpy as np
import skimage.io
import PIL
import matplotlib.pyplot as plt
import cv2
import pickle
import collections
import csv

from cached_property import cached_property
from numpy import transpose
from pip._vendor.html5lib.filters import sanitizer
from itertools import permutations
from matplotlib.dates import SecondLocator

data_path ="/home/eugenio/workspace/documets di pasqualino/Documents/OCTdatabase"
#model_file = os.path.join(data_path, "model.h5")
#M= load_model(model_file)
#print (M.summary())
'''   dato il nome di un paziente il grado dello spicchio e il passo crea i dati degli spicchi
       e li restituisce etichettati da Y restituendo anche i labels del pullback  '''


def pullback_n_labels_part_yeah (env_path,database, p_i,p_f,deg=4,stride=2) :
    
    
    pullback_path = env_path +database 
    labels_path =  env_path+"/selmixedlabels.npy"    
    guide_path=env_path+"/selmixedguide.npy"
    
    pullback=np.load(pullback_path,mmap_mode='r')
    labels=np.load(labels_path)
    guide=np.load(guide_path)
    
    scarto=(pullback.shape[1]-deg)%stride
    
    if (scarto!=0) :
        pullback=pullback[:, 0:-scarto,:]
        labels=labels[:, 0:-scarto]
                
    first_frame=int(pullback.shape[0]/100*p_i)
    last_frame=int(pullback.shape[0]/100*p_f)
    
    nSlices=last_frame-first_frame
    n4Slice= int ((pullback.shape[1]-deg)/stride)
    nTot= n4Slice*nSlices
    X=np.zeros((nTot,deg,pullback.shape[2]))
    Y=np.zeros((nTot,1))
           
       
    count=0             
    count_guide=0
    for i in range(first_frame,last_frame):
        for j in range(n4Slice):
            if(sum(guide[i,j*stride:j*stride+deg])==0):
                count+=1
                X[(i - first_frame)*n4Slice + j,:,:] = pullback[i,j*stride:j*stride+deg,:]
                if (sum(labels[i ,j*stride:j*stride+deg])>deg//2):
                    Y[(i - first_frame)*n4Slice + j-count_guide] = 1
                    #print(sum(labels[i,j*stride:j*stride+deg]))
                else:
                    Y[(i - first_frame)*n4Slice + j-count_guide] = 0
            else:
                count_guide+=1
    X=X[0:count,:,:]
    Y=Y[0:count]
    
    print(nTot)
    print(X.shape)
        
    return X,Y,labels[first_frame:last_frame,:],nSlices,pullback.shape[1]


def pullback_n_labels_part_min (env_path,database, p_i,p_f,depth=300,deg=4,stride=2) :
    
    
    pullback_path = env_path +database 
    labels_path =  env_path+"/selmixedlabels.npy"    
    guide_path=env_path+"/selmixedguide.npy"
    intima_path=env_path+"/selmixedintima.npy"
    
    pullback=np.load(pullback_path,mmap_mode='r')
    labels=np.load(labels_path)
    guide=np.load(guide_path)
    intima=np.load(intima_path)
    
    scarto=(pullback.shape[1]-deg)%stride
    
    if (scarto!=0) :
        pullback=pullback[:, 0:-scarto,:]
        labels=labels[:, 0:-scarto]
                
    first_frame=int(pullback.shape[0]/100*p_i)
    last_frame=int(pullback.shape[0]/100*p_f)
    
    nSlices=last_frame-first_frame
    n4Slice= int ((pullback.shape[1]-deg)/stride)
    nTot= n4Slice*nSlices
    X=np.zeros((nTot,deg,depth))
    Y=np.zeros((nTot,1))
           
    massimo=0   
    count=0             
    count_guide=0
    for i in range(first_frame,last_frame):
        for j in range(n4Slice):
            if(sum(guide[i,j*stride:j*stride+deg])==0):
                count+=1
                m=np.min(intima[i,j*stride:j*stride+deg])
                #print(m)
                #print(pullback[i,j*stride:j*stride+deg,:].shape)
                max_depth= pullback.shape[2]-m
                if(max_depth<depth):
                    X[(i - first_frame)*n4Slice + j-count_guide,:,0:max_depth]=pullback[i,j*stride:j*stride+deg,m:pullback.shape[2]]
                else:
                    X[(i - first_frame)*n4Slice + j-count_guide,:,:]=pullback[i,j*stride:j*stride+deg,m:m+depth]
                if (sum(labels[i ,j*stride:j*stride+deg])>deg//2):
                        Y[(i - first_frame)*n4Slice + j-count_guide] = 1
                        #print(sum(labels[i,j*stride:j*stride+deg]))
                else:
                    Y[(i - first_frame)*n4Slice + j-count_guide] = 0
            else:
                count_guide+=1
    X=X[0:count,:,:]
    Y=Y[0:count]
    
    print(nTot)
    print(X.shape)
    print (massimo)
    return X,Y,labels[first_frame:last_frame,:],nSlices,pullback.shape[1]



def pullback_n_labels_part_min_multiclass (env_path,database, p_i,p_f,classes=3,depth=300,deg=4,stride=2,mixed_mode=False) :
    
    
    pullback_path = env_path +database 
    if(mixed_mode):
        labels_path =  env_path+"/selmixedlabels.npy"
        calcium_path =  env_path+"/selmixedcalcium.npy"
        guide_path=env_path+"/selmixedguide.npy"
        intima_path=env_path+"/selmixedintima.npy"
    else:  
        labels_path =  env_path+"/sellabels.npy"
        calcium_path =  env_path+"/selcalcium.npy"
        guide_path=env_path+"/selguide.npy"
        intima_path=env_path+"/selintima.npy"
        
    pullback=np.load(pullback_path,mmap_mode='r')
    labels=np.load(labels_path)
    calcium=np.load(calcium_path)
    guide=np.load(guide_path)
    intima=np.load(intima_path)
    
    scarto=(pullback.shape[1]-deg)%stride
    
    if (scarto!=0) :
        pullback=pullback[:, 0:-scarto,:]
        labels=labels[:, 0:-scarto]
                
    first_frame=int(pullback.shape[0]/100*p_i)
    last_frame=int(pullback.shape[0]/100*p_f)
    
    nSlices=last_frame-first_frame
    n4Slice= int ((pullback.shape[1])/stride)
    N=int ((pullback.shape[1]-deg)/stride) +1
    nTot= n4Slice*nSlices
    X=np.zeros((nTot,deg,depth))
    Y=np.zeros((nTot,classes))
           
    massimo=0   
    count=0             
    count_guide=0
    for i in range(first_frame,last_frame):
        for j in range(n4Slice):
            if(j<N):
                if(sum(guide[i,j*stride:j*stride+deg])==0):
                    count+=1
                    m=np.min(intima[i,j*stride:j*stride+deg])
                    #print(m)
                    #print(pullback[i,j*stride:j*stride+deg,:].shape)
                    max_depth= pullback.shape[2]-m
                    if(max_depth<depth):
                        X[(i - first_frame)*n4Slice + j-count_guide,:,0:max_depth]=pullback[i,j*stride:j*stride+deg,m:pullback.shape[2]]
                    else:
                        X[(i - first_frame)*n4Slice + j-count_guide,:,:]=pullback[i,j*stride:j*stride+deg,m:m+depth]
                    if (sum(calcium[i ,j*stride:j*stride+deg])>deg//3):
                            Y[(i - first_frame)*n4Slice + j-count_guide,2] = 1
                    elif (sum(labels[i ,j*stride:j*stride+deg])>deg//2):
                            Y[(i - first_frame)*n4Slice + j-count_guide,1] = 1
                    else:
                        Y[(i - first_frame)*n4Slice + j-count_guide,0] = 1
                else:
                    count_guide+=1
            else:
                if( sum(guide[i,j*stride:]) + sum(guide[i,0:(j%N+1)*stride]) == 0 ):
                    count+=1
                    m=np.min(  [ np.min(intima[i,j*stride:]) , np.min(intima[i,0:(j%N+1)*stride]) ]   )
                    #print(m)
                    #print(pullback[i,j*stride:j*stride+deg,:].shape)
                    max_depth= pullback.shape[2]-m
                    if(max_depth<depth):
                        X[(i - first_frame)*n4Slice + j-count_guide,0:deg-(j%N+1)*stride,0:max_depth] = pullback[i,j*stride:,m:pullback.shape[2]] 
                        X[(i - first_frame)*n4Slice + j-count_guide,deg-(j%N+1)*stride:,0:max_depth] = pullback[i,0:(j%N+1)*stride,m:pullback.shape[2]]
                        
                    else:
                        #print(np.concatenate([pullback[i,j*stride:,m:m+depth] , pullback[i,0:(j%N+1)*stride,m:m+depth]],axis=1).shape)
                        X[(i - first_frame)*n4Slice + j-count_guide,0:deg-(j%N+1)*stride,:] = pullback[i,j*stride:,m:m+depth] 
                        X[(i - first_frame)*n4Slice + j-count_guide,deg-(j%N+1)*stride:,:] = pullback[i,0:(j%N+1)*stride,m:m+depth]
                    if (sum(calcium[i,j*stride:]) + sum(calcium[i,0:(j%N+1)*stride])>deg//3):    
                            Y[(i - first_frame)*n4Slice + j-count_guide,2] = 1
                    elif ( sum(labels[i,j*stride:]) + sum(labels[i,0:(j%N+1)*stride]) > deg//2):
                            Y[(i - first_frame)*n4Slice + j-count_guide,1] = 1
                    else:
                        Y[(i - first_frame)*n4Slice + j-count_guide,0] = 1
                else:
                    count_guide+=1
    X=X[0:count,:,:]
    Y=Y[0:count]
    
    print(nTot)
    print(X.shape)
    print (massimo)
    return X,Y,labels[first_frame:last_frame,:],nSlices,pullback.shape[1]



def ricostruisci_labels(Nf,Nr,Y, deg=4, stride=2):
    
    labels=np.zeros([Nf, Nr])
    
    #for i in range(Nf):
    #    for j in range(Nr):
    
            
    for k in range(len(Y)):
        if(Y[k]==1) :
            labels [k//((Nr-deg)//stride),  k%((Nr-deg)//stride)*stride : k%((Nr-deg)//stride)*stride+deg] = 1
        
    return labels


def ricostruisci_labels_continue(Nf,Nr,Y, deg=4, stride=2):
    
    labels=np.zeros([Nf, Nr])
    for k in range(len(Y)):        
        labels [k//((Nr-deg)//stride),  k%((Nr-deg)//stride)*stride : k%((Nr-deg)//stride)*stride+deg] = Y[k]
        
    return labels

def ricostruisci_labels_guide_yeah_yeah(env_path,p_i,p_f,Nf,Nr,Y, deg=4, stride=2,mixed_mode=False):
  
    labels=np.zeros([Nf, Nr])
    
    if(mixed_mode):
        guide=np.load(env_path+"/selmixedguide.npy")
    else:  
        guide=np.load(env_path+"/selguide.npy")
      
    FS=int(guide.shape[0]*((guide.shape[1]-deg)//stride)*p_i/100)
    LS=int(guide.shape[0]*((guide.shape[1]-deg)//stride)*p_f/100)
      
    FF=int(guide.shape[0]/100*p_i)
    LF=int(guide.shape[0]/100*p_f)
    
    nSlices=LF-FF
    n4Slice= int (Nr/stride) 
    N=int ((Nr-deg)/stride) +1
    nTot= n4Slice*nSlices

    count=0
    for k in range(nTot):
        if sum(guide[FF+k//((Nr-deg)//stride),  k%((Nr-deg)//stride)*stride: k%((Nr-deg)//stride)*stride+deg])>0 :
            count+=1
            #print(count) 19501
            if k%((Nr-deg)//stride)==0 :
                labels [k//((Nr-deg)//stride),  k%((Nr-deg)//stride)*stride : k%((Nr-deg)//stride)*stride+deg] = 2
            else :
                labels [k//((Nr-deg)//stride),  k%((Nr-deg)//stride)*stride + deg - stride : k%((Nr-deg)//stride)*stride+deg] = 2
        else :
            if(Y[k-count]==1) :
                labels [k//((Nr-deg)//stride),  k%((Nr-deg)//stride)*stride : k%((Nr-deg)//stride)*stride+deg] = 1
              
    return labels



def build_labels_topology(env_path,p_i,p_f,Nf,Nr,Y, deg=4, stride=2):
  
    T=np.empty((Nf, Nr),dtype=object)
    
    for i in range(Nf):
        for j in range(Nr):
            T[i,j]=[]
    
    print(type(T[0,0]))
    
    guide=np.load(env_path+"/selmixedguide.npy")
      
    FS=int(guide.shape[0]*((guide.shape[1]-deg)//stride)*p_i/100)
    LS=int(guide.shape[0]*((guide.shape[1]-deg)//stride)*p_f/100)
      
    FF=int(guide.shape[0]/100*p_i)
    LF=int(guide.shape[0]/100*p_f)
    
    nSlices=LF-FF
    n4Slice= int ((guide.shape[1]-deg)/stride)
    nTot= n4Slice*nSlices
    
    count=0
    for k in range(nTot):
        if sum(guide[FF+k//((Nr-deg)//stride),  k%((Nr-deg)//stride)*stride: k%((Nr-deg)//stride)*stride+deg])>0 :
            count+=1
            
        else :
            for h in range(k%((Nr-deg)//stride)*stride , k%((Nr-deg)//stride)*stride+deg):
                T [k//((Nr-deg)//stride), h ].append(Y[k-count])
              
    return T


def ricostruisci_labels_guide_multi(env_path,p_i,p_f,Nf,Nr,Y, deg=4, stride=2,mixed_mode=False):
  
    labels=np.zeros([Nf, Nr])
    if(mixed_mode):
        guide=np.load(env_path+"/selmixedguide.npy")
    else:  
        guide=np.load(env_path+"/selguide.npy")
      
    FS=int(guide.shape[0]*((guide.shape[1]-deg)//stride)*p_i/100)
    LS=int(guide.shape[0]*((guide.shape[1]-deg)//stride)*p_f/100)
      
    FF=int(guide.shape[0]/100*p_i)
    LF=int(guide.shape[0]/100*p_f)
    
    nSlices=LF-FF
    n4Slice= int (Nr/stride) 
    N=int ((Nr-deg)/stride) +1
    nTot= n4Slice*nSlices
    count=0
    for k in range(nTot):

        if(k%n4Slice*stride+deg<=Nr):
            if sum(guide[FF+k//n4Slice,  k%n4Slice*stride: k%n4Slice*stride+deg])>0 :
                count+=1
                #print(count) 19501
                #if k%n4Slice ==0 :
                #labels [k//n4Slice,  k%n4Slice*stride : k%n4Slice*stride+deg] = 2
                #else :
                #    labels [k//n4Slice,  k%n4Slice*stride + deg - stride : k%n4Slice*stride+deg] = 2
            else :
                if(Y[k-count]==2) :
                    labels [k//n4Slice,  k%n4Slice*stride : k%n4Slice*stride+deg] = 3
                
                elif(Y[k-count]==1) :
                    labels [k//n4Slice,  k%n4Slice*stride : k%n4Slice*stride+deg] = 1
        else :
            if sum(guide[FF+k//n4Slice,  k%n4Slice*stride:  Nr]) + sum(guide[FF+k//n4Slice, 0 : (k%n4Slice*stride+deg)%Nr ]) > 0 :
                count+=1
                #print(count) 19501
                #if k%n4Slice == 0 :
                #labels [k//n4Slice,  k%n4Slice*stride : Nr] = 2
                #labels [k//n4Slice,  0 : (k%n4Slice*stride+deg)%Nr] = 2
                #else :
                #    labels [k//n4Slice,  k%n4Slice*stride + deg - stride : k%n4Slice*stride+deg] = 2
            else :
                if(Y[k-count]==2) :
                    labels [k//n4Slice,  k%n4Slice*stride : Nr] = 3
                    labels [k//n4Slice,  0 : (k%n4Slice*stride+deg)%Nr] = 3
                
                elif(Y[k-count]==1) :
                    labels [k//n4Slice,  k%n4Slice*stride : Nr] = 1
                    labels [k//n4Slice,  0 : (k%n4Slice*stride+deg)%Nr] = 1
                
    return labels


'''

def ricostruisci_labels_guide_multi(env_path,p_i,p_f,Nf,Nr,Y, deg=4, stride=2,mixed_mode=False):
  
    labels=np.zeros([Nf, Nr])
    if(mixed_mode):
        guide=np.load(env_path+"/selmixedguide.npy")
    else:  
        guide=np.load(env_path+"/selguide.npy")
      
    FS=int(guide.shape[0]*((guide.shape[1]-deg)//stride)*p_i/100)
    LS=int(guide.shape[0]*((guide.shape[1]-deg)//stride)*p_f/100)
      
    FF=int(guide.shape[0]/100*p_i)
    LF=int(guide.shape[0]/100*p_f)
    
    nSlices=LF-FF
    n4Slice= int ((guide.shape[1]-deg)/stride)
    nTot= n4Slice*nSlices
    print(Y.shape)
    count=0
    for k in range(nTot):
        if sum(guide[FF+k//((Nr-deg)//stride),  k%((Nr-deg)//stride)*stride: k%((Nr-deg)//stride)*stride+deg])>0 :
            count+=1
            #print(count) 19501
            if k%((Nr-deg)//stride)==0 :
                labels [k//((Nr-deg)//stride),  k%((Nr-deg)//stride)*stride : k%((Nr-deg)//stride)*stride+deg] = 2
            else :
                labels [k//((Nr-deg)//stride),  k%((Nr-deg)//stride)*stride + deg - stride : k%((Nr-deg)//stride)*stride+deg] = 2
        else :
            if(Y[k-count]==2) :
                labels [k//((Nr-deg)//stride),  k%((Nr-deg)//stride)*stride : k%((Nr-deg)//stride)*stride+deg] = 3
            
            elif(Y[k-count]==1) :
                labels [k//((Nr-deg)//stride),  k%((Nr-deg)//stride)*stride : k%((Nr-deg)//stride)*stride+deg] = 1
              
    return labels

'''

    
def ricostruisci_labels_continue_guida(env_path,p_i,p_f,Nf,Nr,Y, deg=4, stride=2):
    
    labels=np.zeros([Nf, Nr])
    guide=np.load(env_path+"/selmixedguide.npy")
      
    FS=int(guide.shape[0]*((guide.shape[1]-deg)//stride)*p_i/100)
    LS=int(guide.shape[0]*((guide.shape[1]-deg)//stride)*p_f/100)
      
    FF=int(guide.shape[0]/100*p_i)
    LF=int(guide.shape[0]/100*p_f)
    
    nSlices=LF-FF
    n4Slice= int ((guide.shape[1]-deg)/stride)
    nTot= n4Slice*nSlices
    
    count=0
    for k in range(nTot):
        if sum(guide[FF+k//((Nr-deg)//stride),  k%((Nr-deg)//stride)*stride: k%((Nr-deg)//stride)*stride+deg])>0 :
            count+=1
            #print(count) 19501
            if k%((Nr-deg)//stride)==0 :
                labels [k//((Nr-deg)//stride),  k%((Nr-deg)//stride)*stride : k%((Nr-deg)//stride)*stride+deg] = 2
            else :
                labels [k//((Nr-deg)//stride),  k%((Nr-deg)//stride)*stride + deg - stride : k%((Nr-deg)//stride)*stride+deg] = 2
        else :
            labels [k//((Nr-deg)//stride),  k%((Nr-deg)//stride)*stride : k%((Nr-deg)//stride)*stride+deg] = Y[k-count]  
        
    return labels


    
def ricostruisci_labels_mean_guida(env_path,p_i,p_f,Nf,Nr,Y, thr, deg=4, stride=2):
    
    labels=np.zeros([Nf, Nr])
    mean_counter=np.zeros([Nf, Nr])
    guide=np.load(env_path+"/selmixedguide.npy")
      
    FS=int(guide.shape[0]*((guide.shape[1]-deg)//stride)*p_i/100)
    LS=int(guide.shape[0]*((guide.shape[1]-deg)//stride)*p_f/100)
      
    FF=int(guide.shape[0]/100*p_i)
    LF=int(guide.shape[0]/100*p_f)
    
    nSlices=LF-FF
    n4Slice= int ((guide.shape[1]-deg)/stride)
    nTot= n4Slice*nSlices
    
    count=0
    for k in range(nTot):
        if sum(guide[FF+k//((Nr-deg)//stride),  k%((Nr-deg)//stride)*stride: k%((Nr-deg)//stride)*stride+deg])>0 :
            count+=1
            #print(count) 19501
            if k%((Nr-deg)//stride)==0 :
                labels [k//((Nr-deg)//stride),  k%((Nr-deg)//stride)*stride : k%((Nr-deg)//stride)*stride+deg] = 2
            else :
                labels [k//((Nr-deg)//stride),  k%((Nr-deg)//stride)*stride + deg - stride : k%((Nr-deg)//stride)*stride+deg] = 2
        else :
            labels [k//((Nr-deg)//stride),  k%((Nr-deg)//stride)*stride : k%((Nr-deg)//stride)*stride+deg] += Y[k-count]  
        
            mean_counter [k//((Nr-deg)//stride),  k%((Nr-deg)//stride)*stride : k%((Nr-deg)//stride)*stride+deg] += 1  
        
        
    for i in range(Nf):
        for j in range(Nr):
            if mean_counter[i,j]!=0:
                labels[i,j]/=mean_counter[i,j]
                if labels[i,j]>thr:
                    labels[i,j]=1
                else:
                    labels[i,j]=0
                    
    
    return labels



def plot_order(ann_path,p_i,p_f):

    mixed_labels=np.load(ann_path)#,mmap_mode='r')
    FF=int(mixed_labels.shape[0]*p_i/100)
    LF=int(mixed_labels.shape[0]*p_f/100)
    labels=mixed_labels[FF:LF,:].astype(np.uint16)
    #labels = labels.astype(np.uint16)
    longness=np.sum(labels,axis=1)
    
    ordering=[]
    for i in range(labels.shape[0]):
        ordering.append((longness[i],i))
    ordering=sorted(ordering, key=lambda i: i[0])
    
    permutation=[]
    for i in range(labels.shape[0]):
        permutation.append(ordering[i][1])
    permutation=np.array(permutation,dtype=np.int64)
    
    '''ind=lambda frame: frame[1]
    permutation=np.array(ind(ordering),dtype=np.int32)
    print(ind(ordering))
    print(permutation.shape)'''
    
    labels = labels[permutation,:]
    begins=-np.ones((labels.shape[0]))
    for i in range(labels.shape[0]):
        j=0
        pre=1
        while j<labels.shape[1] :
            if(labels[i,j]==1 and pre==0):
                begins[i]=j
                j=labels.shape[1]
            else :
                pre=labels[i,j]
                j+=1
        if(begins[i]==-1 and labels[i,0]==1):
            begins[i]=0
    
    i=0
    i0=0
    i1=0
    i2=0
    i3=0
    
    while(ordering[i][0]==0):
        i+=1
    i0=i
    
    while(ordering[i][0]<100):
        i+=1
    i1=i
    
    while(ordering[i][0]<200):
        i+=1
    i2=i
    
    while(ordering[i][0]<300):
        i+=1
    i3=i
    
    
    final_ordering=[]
    for i in range(i0):
        final_ordering.append((begins[i],permutation[i]))
    
    ordering=[]
    for i in range(i0,i1):
        ordering.append((begins[i],permutation[i]))
    ordering=sorted(ordering, key=lambda i: i[0])
    for i in range(i1-i0):
        final_ordering.append(ordering[i])
    
    ordering=[]
    for i in range(i1,i2):
        ordering.append((begins[i],permutation[i]))
    ordering=sorted(ordering, key=lambda i: i[0])
    for i in range(i2-i1):
        final_ordering.append(ordering[i])
    
    ordering=[]
    for i in range(i2,i3):
        ordering.append((begins[i],permutation[i]))
    ordering=sorted(ordering, key=lambda i: i[0])
    for i in range(i3-i2):
        final_ordering.append(ordering[i])
    
    ordering=[]
    for i in range(i1,labels.shape[0]):
        ordering.append((begins[i],permutation[i]))
    ordering=sorted(ordering, key=lambda i: i[0])
    for i in range(labels.shape[0]-i3):
        final_ordering.append(ordering[i])
    
    
    permutation=[]
    for i in range(labels.shape[0]):
        permutation.append(final_ordering[i][1])
    permutation=np.array(permutation,dtype=np.int64)
    
    
    print (len(final_ordering))#[0:100])
    print(labels.shape)
    print(np.max(permutation))
    
    return permutation

def pred_to_label(preds, thr=0.5):
    l=np.zeros(preds.shape)
    for i in range(len(preds)) : 
        if (preds[i]>thr) :
            l[i]=1
        else :
            l[i]=0
        
    return l

def pred_to_label_multiclass(preds, thr=0.5,thr_ca=0.5):
    l=np.zeros(preds.shape[0])
    
    l=np.argmax(preds, axis=1)
    print(l.shape)
    
    for i in range(preds.shape[0]) : 
        THR=thr
        if l[i]==2:
            THR=thr_ca
        if (preds[i,l[i]]<THR) :
            l[i]=0
    return l

    

def pred_to_label_multiclass2(preds, thr=0.5,thr_ca=0.5):
    l=np.zeros(preds.shape[0])
    
    l=np.argmax(preds, axis=1)
    
    THR=np.zeros(3)
    THR[0]=0.
    THR[1]=thr
    THR[2]=thr_ca
    for i in range(preds.shape[0]) : 
        
        if (preds[i,l[i]]<THR[l[i]]) :
            s=np.argmax(preds[i,np.delete([0,1,2],l[i])])

            if (s!=0) : 
                if preds[i,np.delete([0,1,2],np.array([0,l[i]]))]>THR[np.delete([0,1,2],np.array([0,l[i]]))]:
                    l[i]=np.delete([0,1,2],[0,l[i]])
            else:
                l[i]=0
     
    return l



def pred_to_label_multiclass3(preds, thr=0.5,thr_ca=0.5):
    l=np.zeros(preds.shape[0])
    
    l=np.argmax(preds, axis=1)
    
    THR=np.zeros(3)
    THR[0]=0.
    THR[1]=thr
    THR[2]=thr_ca
    for i in range(preds.shape[0]) : 
        
        if (preds[i,l[i]]<THR[l[i]]) :
            s=np.argmax(preds[i,np.delete([0,1,2],l[i])])

            if (s!=0) : 
                if preds[i,np.delete([0,1,2],np.array([0,l[i]]))]>THR[np.delete([0,1,2],np.array([0,l[i]]))]:
                    l[i]=np.delete([0,1,2],[0,l[i]])
            else:
                l[i]=0
     
    return l


def pred_to_label_1VSall(preds_te,c, thr):
    if(c==0):
        l=np.ones(preds_te.shape[0])
    else :
        l=np.zeros(preds_te.shape[0])
    for i in range(preds_te.shape[0]) : 
        if (preds_te[i,c]>thr):
            l[i]=c
    return l

def binpred_to_multipred(preds_te):
    pred_multi=np.zeros((preds_te.shape[0],2))

    for i in range(preds_te.shape[0]):
        pred_multi[i,0]=1-preds_te[i]
        pred_multi[i,1]=preds_te[i]
    return pred_multi

def multipred_to_binpred(preds_multi,c):

    return pred_multi[c]

def mono_labels(Y,nc):
    mY=np.zeros(Y.shape[0])
    for i in range (Y.shape[0]):
        for j in range(nc):
            if Y[i,j]==1:
                mY[i]=j
    
    return mY

def save_list(l,model_path):
    with open(model_path, "wb") as fp:   #Pickling
        pickle.dump(l, fp)
    return

def load_list(model_path):
    with open(model_path, "rb") as fp:   # Unpickling
        b = pickle.load(fp)
    return b  


def selesao(env_path):
    
    #
    #bad=load_list("/home/eugenio/workspace/documets di pasqualino/Documents/pasqualino/placcaultras/bad.txt")
    bad=load_list("/home/fisici/Documents/OperazionePlacca/bad.txt")

    bad=clean_list(bad)
    
    pullback_path = env_path + "/allpullbacks.npy" 
    
    #labels_path =  env_path+"/alllabels.npy"    
    #guide_path=env_path+"/allguide.npy"
    #intima_path=env_path+"/allintima.npy"
    
    #labels_path = "/home/eugenio/workspace/documets di pasqualino/Documents/pasqualino/placcaultras/alllabelsnew.npy"    
    #guide_path="/home/eugenio/workspace/documets di pasqualino/Documents/pasqualino/placcaultras/allartifactnew.npy"
    #calcium_path="/home/eugenio/workspace/documets di pasqualino/Documents/pasqualino/placcaultras/allcalciumnew.npy"
    #intima_path="/home/eugenio/workspace/documets di pasqualino/Documents/pasqualino/placcaultras/allintimanew.npy"
    
    labels_path = "/home/fisici/Documents/OperazionePlacca/alllabelsnew.npy"    
    guide_path="/home/fisici/Documents/OperazionePlacca/allartifactnew.npy"
    calcium_path="/home/fisici/Documents/OperazionePlacca/allcalciumnew.npy"
    intima_path="/home/fisici/Documents/OperazionePlacca/allintimanew.npy"
    
    pullback=np.load(pullback_path,mmap_mode='r')
    labels=np.load(labels_path)
    guide=np.load(guide_path)
    calcium=np.load(calcium_path)
    intima=np.load(intima_path)
    
    buoni=np.arange(intima.shape[0])
    bad=np.array(bad)
    buoni=np.delete(buoni,bad)
    
    np.save(env_path+"/selpullbacks.npy",pullback[buoni,:,:])
    print("pullback fatto")
    np.save(env_path+"/sellabels.npy",labels[buoni,:])
    
    np.save(env_path+"/selguide.npy",guide[buoni,:])
    
    np.save(env_path+"/selintima.npy",intima[buoni,:])
    
    np.save(env_path+"/selcalcium.npy",calcium[buoni,:])
    
    
    return


def cut_100(env_path):
    
    bad=load_list("/home/eugenio/workspace/documets di pasqualino/Documents/pasqualino/placcaultras/bad.txt")

    pullback_path = env_path + "/selmixedpullbacks.npy" 
    
    pullback=np.load(pullback_path,mmap_mode='r')
    
    np.save(env_path+"/selcutmixedpullbacks.npy",pullback[:,:,100:-1])
    
    return


class _OrderedDictMaker(object):
    def __getitem__(self, keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        assert all(isinstance(key, slice) for key in keys)

        return collections.OrderedDict([(k.start, k.stop) for k in keys])

def WriteOrdDictToCSV(csv_file,Info):    
    keys, values = [], []
    
    for key, value in Info.items():
        keys.append(key)
        values.append(value)       
    
    with open(csv_file, "a") as outfile:
        csvwriter = csv.writer(outfile)
        if os.stat(csv_file).st_size == 0 :
            csvwriter.writerow(keys)
        csvwriter.writerow(values)
    return


def error_matrix(preds_te,Y_test,nc):
                
    error_matrix=np.zeros((nc,nc))
    tot4class=np.zeros(nc)
    for i in range(len(preds_te)) :
        error_matrix[preds_te[i],Y_test[i]]+=1
        tot4class[preds_te[i]]+=1
    error_matrix=np.divide(np.transpose(error_matrix),tot4class)
    
    
    return error_matrix



def roc_multiclass(model_path, preds_multi_te,Y_test,nc,tot_thr):
    
    ROC=[]
    scores=np.zeros((11,nc, tot_thr+1))
    
    if(nc==2):
        preds_multi_te=DP.binpred_to_multipred(preds_multi_te)
    if(nc>2):
        Y_test=mono_labels(Y_test,nc)
    for c in range(nc) :
        ROC_x=[]
        ROC_y=[]
        print(c)
        for n in range(tot_thr+1):        
            #preds_te = DP.pred_to_label_multiclass2(preds_te, thr=1./20*n,thr_ca=0.7)#Info['threshold'],thr_ca=Info['threshold_ca'])
            preds_te = pred_to_label_1VSall(preds_multi_te,c, thr=(1-1./tot_thr*n)**3)#1./tot_thr*n)#
            
            preds_te = preds_te.astype(np.uint16)
            Y_test=Y_test.astype(np.uint16)
            
            print(preds_te.shape)
            print(Y_test.shape)
            for i in range(len(preds_te)) :
                if (preds_te[i] == Y_test[i] ) :
                    if(preds_te[i] == c):
                        scores[0,c,n] = scores[0,c,n] + 1 #TP
                    else : 
                        scores[2,c,n] = scores[2,c,n] + 1 #TN
                else :
                    if(preds_te[i] == c):
                        scores[1,c,n] = scores[1,c,n] + 1 #FP
                    else: 
                        scores[3,c,n] = scores[3,c,n] + 1 #FN
            #print(scores[0:4,:])
            if scores[0,c,n]+scores[1,c,n]>0:
                scores[4,c,n]=np.divide(scores[0,c,n],(scores[0,c,n]+scores[1,c,n])) #precision(TP tra i P, percentuale di placca tra quella rilevata)
            if scores[0,c,n]+scores[3,c,n]>0:
                scores[5,c,n]=np.divide(scores[0,c,n],(scores[0,c,n]+scores[3,c,n])) #recall   (placca trovata ta quella esistente) (TPR)
            if scores[1,c,n]+scores[2,c,n]>0:
                scores[6,c,n]=np.divide(scores[1,c,n],(scores[1,c,n]+scores[2,c,n])) #FPR
        
            scores[7,c,n]=2*np.divide(np.multiply(scores[4,c,n],scores[5,c,n]),(scores[4,c,n]+scores[5,c,n])) #f1
            
            scores[8,c,n]=np.divide(scores[5,c,n],scores[6,c,n]) #PL
            scores[9,c,n]=np.divide(1-scores[5,c,n],1-scores[6,c,n]) #NL
            scores[10,c,n]=np.divide(scores[8,c,n],scores[9,c,n]) #DOR
            
            ROC_x.append(scores[6,c,n])
            
            ROC_y.append(scores[5,c,n])
        #print(scores[:,c,1:100:10])
        plt.plot(ROC_x,ROC_y, lw=1)# ROC_x,ROC_y,'bo',  lw=2, markersize=1 )#'x')#
        plt.plot(ROC_x[1:100:3],ROC_y[1:100:3],'bo',  lw=2, markersize=1)
    #plt.plot(np.arange(100)/100,np.arange(100)/100,lw=1)
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.legend(['normal', 'lipid','calcium'], loc='upper left')
    plt.show()
    plt.savefig(model_path + '/ROC_curve.png', bbox_inches='tight')
    np.save(model_path + "/scores.npy", scores)
    #print(scores)
    return scores

def best_f1_mean(scores):
    
    max_f1=0
    best=0
    for i in range(scores.shape[2]):
        #print(np.mean(scores[7,:,i],axis=0))
        #print(scores[7,:,i])
        if np.mean(scores[7,:,i],axis=0)>max_f1 : 
            best=i
            max_f1=np.mean(scores[7,:,i])
            print(max_f1)
            print (i)
    print( scores[:,:,best])
    print((1-1./100*best)**3)
    return


def best_f1s(scores):
    
    for c in range(scores.shape[1]):
        max_f1=0
        best=0
        for i in range(scores.shape[2]):
            #print(np.mean(scores[7,:,i],axis=0))
            #print(scores[7,:,i])
            if scores[7,c,i]>max_f1 and np.isscalar(scores[7,c,i]) and (1-1./100*i)**3>0.5: 
                best=i
                max_f1=scores[7,c,i]
                #print(max_f1)
                #print (i)
        print(scores[:,:,best])
        print((1-1./100*best)**3)
    return

def best_dor(scores):
    
    for c in range(scores.shape[1]):
        max_dor=0
        best=0
        for i in range(scores.shape[2]):
            #print(np.mean(scores[7,:,i],axis=0))
            #print(scores[7,:,i])
            if scores[10,c,i]>max_dor and np.isfinite(scores[10,c,i]) and (1-1./100*i)**3>0.5: 
                best=i
                max_dor=scores[10,c,i]
                #print(max_f1)
                #print (i)
        print(scores[:,:,best])
        print((1-1./100*best)**3)
    return



def evaluate_reconstruction(env_path, model_path, Info, preds,labels_test,nSlices, nRadius,nc,tot_thr):

    scores=np.zeros((11,tot_thr))
    best_f1=0
    best_thr=0

    for n in range(tot_thr):        
        
        if nc > 2:
            preds_te = pred_to_label_1VSall(preds,1, thr=1./tot_thr*n)
        
        labels_pred = ricostruisci_labels_guide_multi(env_path,Info['cv_set'][0],Info['cv_set'][1],nSlices, nRadius, preds_te, deg=Info['degree'], stride=Info['stride'] )
        #labels_pred = DP.ricostruisci_labels(nSlices, nRadius, preds_te, deg=Info['degree'], stride=Info['stride'] ) 
        
        labels_pred = labels_pred.astype(np.uint16)
        
        labels_test = labels_test.astype(np.uint16)

        
        for i in range(labels_test.shape[0]) :
            for j in range(labels_test.shape[1]) : 
                if (labels_pred[i,j] == labels_test[i,j] ) :
                    if(labels_pred[i,j] == 0):
                        scores[2,n] = scores[2,n] + 1 #TN
                    else : 
                        scores[0,n] = scores[0,n] + 1 #TP                        
                else :
                    if(labels_pred[i,j] == 0):
                        scores[3,n] = scores[3,n] + 1 #FN
                    elif (labels_pred[i,j] == 1): 
                        scores[1,n] = scores[1,n] + 1 #FP           
        if scores[0,n]+scores[1,n]>0:
            scores[4,n]=np.divide(scores[0,n],(scores[0,n]+scores[1,n])) #precision(TP tra i P, percentuale di placca tra quella rilevata)
        if scores[0,n]+scores[3,n]>0:
            scores[5,n]=np.divide(scores[0,n],(scores[0,n]+scores[3,n])) #recall   (placca trovata ta quella esistente) (TPR)
        if scores[1,n]+scores[2,n]>0:
            scores[6,n]=np.divide(scores[1,n],(scores[1,n]+scores[2,n])) #FPR
    
        scores[7,n]=2*np.divide(np.multiply(scores[4,n],scores[5,n]),(scores[4,n]+scores[5,n])) #f1
        
        scores[8,n]=np.divide(scores[5,n],scores[6,n]) #PL
        scores[9,n]=np.divide(1-scores[5,n],1-scores[6,n]) #NL
        scores[10,n]=np.divide(scores[8,n],scores[9,n]) #DOR

        if(scores[7,n]>best_f1):
            best_f1=scores[7,n]
            best_thr= thr=1./tot_thr*n
            Info['threshold'] = best_thr 
            Info['TP'] = scores[0,n]
            Info['FP'] = scores[1,n]
            Info['TN'] = scores[2,n]
            Info['FN'] = scores[3,n]
            Info['precision'] = scores[4,n] 
            Info['recall'] = scores[5,n] 
            Info['f1'] = scores[7,n]
                    
        
    print(Info)
    np.save(model_path + "/scores_rec.npy", scores)
    return Info ,scores





def train_indexes():
    
    b=load_list("/home/eugenio/workspace/documets di pasqualino/Documents/OCTdatabase/all_pullbacks/old selsao/bad.txt")
    #b=load_list("/home/fisici/Documents/OCTdatabase/all_pullbacks/old selsao/bad.txt")
    
    
    p=np.load("/home/eugenio/workspace/documets di pasqualino/Documents/OCTdatabase/all_pullbacks/permutation.npy")
    #p=np.load("/home/fisici/Documents/OCTdatabase/all_pullbacks/old selsao/permutation.npy")
    
    count=0
    
    tp=[]
    
    for i in range(p.shape[0]):
        if i in b:
            count+=1
        else:
             tp.append(p[i])
    
    tp=np.array(tp)
    tp=tp[0:int(tp.shape[0]*0.75)]
    tp=np.sort(tp)
    return tp


def bad_count():
    
    b=load_list("/home/eugenio/workspace/documets di pasqualino/Documents/pasqualino/placcaultras/bad.txt")
    p=np.load("/home/eugenio/workspace/documets di pasqualino/Documents/OCTdatabase/all_pullbacks/allcalcium.npy",mmap_mode='r')
    #b=load_list("/home/fisici/Documents/OperazionePlacca/bad.txt")
    #p=np.load("/home/fisici/Documents/OCTdatabase/all_pullbacks/allcalcium.npy",mmap_mode='r')
    
    b=clean_list(b)
    
    count=0
    bc=np.zeros(p.shape[0])
    for i in range(p.shape[0]):
        if i in b:
            count+=1
        bc[i]=count
    
    return bc


def train_indexes_gui():
    
    b=load_list("/home/eugenio/workspace/documets di pasqualino/Documents/pasqualino/placcaultras/bad.txt")
    #b=load_list("/home/fisici/Documents/OperazionePlacca/bad.txt")

    b=clean_list(b)
    
    tp=train_indexes()
    c=0
    bc=bad_count()
    tg=[]
    for i in range(tp.shape[0]):
        if tp[i] in b:
            c+=1
        else:
            tg.append(tp[i]-bc[tp[i]])
  
    return tg 

def clean_list(bad):

    bad.sort()
    i=0
    while(i>=0):
        if(i+1==len(bad)):
             i=-1
        else:
            if(bad[i]==bad[i+1]):
                del bad[i]
                print(i)
            else:
                i+=1
    return bad

#selesao("/home/fisici/Documents/OCTdatabase/all_pullbacks/new data")

'''
model_path="/home/eugenio/workspace/documets di pasqualino/Documents/pasqualino/4.7-1"

P1=np.load(model_path+"/Validation_prediction_new11.npy")
P2=np.load(model_path+"/Validation_prediction_new12.npy")
P3=np.load(model_path+"/Validation_prediction_new13.npy")
P=np.concatenate([P1,P2,P3],axis=0)
print(P.shape)
np.save(model_path +"/validation_prediction_new.npy",P)
'''

'''
selesao("/home/fisici/Documents/OCTdatabase/all_pullbacks")
selesao("/home/eugenio/workspace/documets di pasqualino/Documents/OCTdatabase/all_pullbacks")
a=np.load("/home/eugenio/workspace/documets di pasqualino/Documents/pasqualino/placcaultras/v.npy",mmap_mode='r+')
#a=a[0:5,:,:]
print(a.shape)
np.save("/home/eugenio/workspace/documets di pasqualino/Documents/pasqualino/placcaultras/v.npy",a)
l=np.load("/home/eugenio/workspace/documets di pasqualino/Documents/pasqualino/placcaultras/alllabelsnew.npy")
bad=load_list("/home/eugenio/workspace/documets di pasqualino/Documents/pasqualino/placcaultras/bad.txt")

sl=np.load("/home/eugenio/workspace/documets di pasqualino/Documents/OCTdatabase/all_pullbacks/sellabels.npy")

print(l.shape[0])

print(sl.shape[0])

FF=int(l.shape[0]*40/100) - len(bad)

print(FF)

print(FF/sl.shape[0]*100)

FF=int(sl.shape[0]*37.2/100)
print(FF)


bad=load_list("/home/eugenio/workspace/documets di pasqualino/Documents/pasqualino/placcaultras/bad.txt")
bad.sort()

print(len(bad))
for i in range(len(bad)-2):
    if(bad[i]==bad[i+1]):
        del bad[i]
for i in range(len(bad)-1):
    if(bad[i]==bad[i+1]):
        print(bad[i])
print(bad)

print(len(bad))

save_list(bad, "/home/eugenio/workspace/documets di pasqualino/Documents/pasqualino/placcaultras/bad.txt")

preds =pred_to_label_multiclass(np.load("/home/eugenio/workspace/documets di pasqualino/Documents/pasqualino/placcaultras/Validation_prediction.npy"))
count=0
for i in range(preds.shape[0]) :
    if preds[i]==2:#(preds[i,2]>0.5):
        count+=1
print(count)
#print(np.max(preds,axis=0))
selesao("/home/eugenio/workspace/documets di pasqualino/Documents/OCTdatabase/all_pullbacks")
X,Y,lab,n1,n2=pullback_n_labels_part_min ("/home/eugenio/workspace/documets di pasqualino/Documents/OCTdatabase/all_pullbacks","/selmixedpullbacks.npy", 0,1,depth=300,deg=64,stride=32)
X1=np.load("/home/eugenio/workspace/documets di pasqualino/Documents/OCTdatabase/all_pullbacks/selmixedpullbacks.npy",mmap_mode='r')

for i in range(X.shape[0]):
    
    F=X[i,:,:]
    F1=X1[0,:,:]
    if(i>10):
        F1=X1[1,:,:]
    
        
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(F)
    axarr[1].imshow(F1)
    
    plt.show()

#cut_100("/home/eugenio/workspace/documets di pasqualino/Documents/OCTdatabase/all_pullbacks")
I=np.load("/home/eugenio/workspace/documets di pasqualino/Documents/OCTdatabase/Aladin_2016_02_19_11_37_21/expertAnnotation/intimaStart.npy")
X=skimage.io.imread("/home/eugenio/workspace/documets di pasqualino/Documents/OCTdatabase/Aladin_2016_02_19_11_37_21/pullback.oct")
#F=np.zeros(X[1,:,:].shape)
for j in range(30):    
    F=X[j,:,:]
    m=np.max(F)
    for k in range(X.shape[1]):
        F[k,I[k,j]]=m
    F=F/m
    print(I[:,j])
    plt.imshow(F)
    plt.show()

X=np.load("/home/eugenio/workspace/documets di pasqualino/Documents/OCTdatabase/all_pullbacks/shortpullbacks.npy",mmap_mode='r')

Z=np.load("/home/eugenio/workspace/documets di pasqualino/Documents/OCTdatabase/all_pullbacks/allshortnuovi.npy",mmap_mode='r')

for i in range(30):
    F=X[i,:,:]
    F1=Z[i,:,:]
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(F)
    axarr[1].imshow(F1)
    
    plt.show()

X1=np.load("/home/eugenio/workspace/documets di pasqualino/Documents/OCTdatabase/all_pullbacks/mixedpullbacks.npy",mmap_mode='r')

X2=np.load("/home/eugenio/workspace/documets di pasqualino/Documents/OCTdatabase/all_pullbacks/mixedshortnuovi.npy",mmap_mode='r')


for i in range(30):
    F=X[i,:,:]
    F1=X2[0,:,:]
    F2=X1[0,:,:]
    f, axarr = plt.subplots(1,3)
    axarr[0].imshow(F)
    axarr[1].imshow(F1)
    axarr[2].imshow(F2)
    
    plt.show()

I=np.load("/home/eugenio/workspace/documets di pasqualino/Documents/OCTdatabase/all_pullbacks/mixedintima.npy",mmap_mode='r')
print()
'''
#X,_,_,_,_=pullback_n_labels_part_min ("/home/eugenio/workspace/documets di pasqualino/Documents/OCTdatabase/all_pullbacks","/mixedshortnuovi.npy", 0,100,deg=16,stride=8) 
