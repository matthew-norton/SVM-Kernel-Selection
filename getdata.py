import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

################################################################################################################################################      

################################################################################################################################################      

def get_cleveland_heart():
    
    data=pd.read_csv('processed.cleveland.data.csv',header=None)
    data=data.convert_objects(convert_numeric=True)
    
    for i in range(data.shape[1]):
        data[i]=data[i].replace('nan',data[i].mean())
        
    data[13]=data[13].replace(0,-1)
    data[13]=data[13].replace([1,2,3,4],1)
    data=np.array(data,dtype=float)
    
    return [np.fliplr(data[0:200]),np.fliplr(data[200:])]
    
def get_problem2(num_vars=None,splined=None):
    
    if splined==True and num_vars==None:
        
        dataTrain= np.loadtxt('problem#2DataTrainSplined.csv',delimiter=',')
        np.place( dataTrain[:,-1],dataTrain[:,-1]==0,-1)
        dataTrain=np.hstack( (dataTrain[:,-1].reshape(110,1),dataTrain[:,2:-1]))
        
        dataTest=np.loadtxt('problem#2DataTestSplined.csv',delimiter=',')
        np.place( dataTest[:,-1],dataTest[:,-1]==0,-1)
        dataTest=np.hstack( (dataTest[:,-1].reshape(168,1),dataTest[:,2:-1]))
        
        return [dataTrain,dataTest]
    
    elif num_vars==None: return [np.loadtxt('problem#2DataTrain.txt').T,np.loadtxt('problem#2DataTest.txt').T]
    elif num_vars==105: return [np.loadtxt('problem#2Data105vars.txt')[0:110],np.loadtxt('problem#2Data105vars.txt')[110:]]
    elif num_vars==14: return [np.loadtxt('problem#2Data14vars.txt')[0:110],np.loadtxt('problem#2Data14vars.txt')[110:]]
    elif num_vars==7: return [np.loadtxt('problem#2Data7vars.txt')[0:110],np.loadtxt('problem#2Data7vars.txt')[110:]]
    elif num_vars==5: return [np.loadtxt('problem#2Data5vars.txt')[0:110],np.loadtxt('problem#2Data5vars.txt')[110:]]

def get_bankdata2(num_chunks,chunksize):
    
    datachunker=pd.read_csv('bankdata2Train.csv',header=None,chunksize=chunksize,dtype=float)
    count=0
    for chunk in datachunker:
        count+=1
        print count
        if count==1: dataTrain=chunk
        elif count<=num_chunks: dataTrain=np.concatenate((dataTrain,chunk))
    
    datachunker=pd.read_csv('bankdata2Test.csv',header=None,chunksize=chunksize)
    count=0
    for chunk in datachunker:
        count+=1
        print count
        if count==1: dataTest=chunk
        elif count<=num_chunks: dataTest=np.concatenate((dataTest,chunk))
    dataTrain=normalize(np.array(dataTrain),axis=0)
    dataTest=normalize(np.array(dataTest),axis=0)
    
    return [dataTrain,dataTest]

def get_ExtraNeuro():
    return np.loadtxt('ExtraNeuro.csv',delimiter=',')
    

def get_ionosphere():
    return [np.loadtxt('ionosphere.txt',delimiter='\t')[0:240],np.loadtxt('ionosphere.txt')[240:]]
    
def get_german():
    return [np.fliplr(np.loadtxt('german.csv',delimiter=',')[0:750]),np.fliplr(np.loadtxt('german.csv',delimiter=',')[750:])]

def get_iris(catagory):
    
    data=np.loadtxt('iris.txt')
    np.place(data[:,0],data[:,0]!=catagory,-1)
    np.place(data[:,0],data[:,0]==catagory,1)
    
    return [np.concatenate((data[0:25],data[75:125])), np.concatenate((data[25:75],data[125:150]))]

def get_digits(digit=0,chunksize=1000,in_chunks=1, out_chunks=1):
    
    datachunker=pd.read_csv('digitTrain.txt',chunksize=chunksize,dtype=float)
    count=0
    for chunks in datachunker:
        count+=1
        if count>in_chunks+out_chunks: break
        elif count==1: dataTrain=np.array(chunks)
        elif 1<count<=in_chunks: dataTrain=np.vstack((dataTrain,np.array(chunks)))
        elif count==1+in_chunks: dataTest=np.array(chunks)
        elif 1+in_chunks<count<=in_chunks+out_chunks: dataTest=np.vstack((dataTest,np.array(chunks)))
        
    np.place(dataTrain[:,0],dataTrain[:,0]!=digit,-1)
    np.place(dataTrain[:,0],dataTrain[:,0]==digit,1)
    np.place(dataTest[:,0],dataTest[:,0]!=digit,-1)
    np.place(dataTest[:,0],dataTest[:,0]==digit,1)

    return [dataTrain,dataTest]

def get_test_circle(noise):
    from numpy.random import random as r

    noise_factor=noise
    data=np.ones((200,3))
    for i in range(50):
        x=(i-25)/float(25)
        data[i,0]=-1;data[i,1]=x*(noise_factor+(1-noise_factor)*2*r());data[i,2]=np.sqrt(1.01-x**2)*(noise_factor+(1-noise_factor)*2*r())
    for i in range(50):
        x=(i-25)/float(25/2)
        data[i+100,0]=1;data[i+100,1]=x*(noise_factor+(1-noise_factor)*2*r());data[i+100,2]=np.sqrt(4.5-x**2)*(noise_factor+(1-noise_factor)*2*r())
    for i in range(50):
        x=(i-25)/float(25)
        data[i+50,0]=-1;data[i+50,1]=x*(noise_factor+(1-noise_factor)*2*r());data[i+50,2]=-np.sqrt(1.01-x**2)*(noise_factor+(1-noise_factor)*2*r())
    for i in range(50):
        x=(i-25)/float(25/2)
        data[i+150,0]=1;data[i+150,1]=x*(noise_factor+(1-noise_factor)*2*r());data[i+150,2]=-np.sqrt(4.5-x**2)*(noise_factor+(1-noise_factor)*2*r())
    return data