import numpy as np
import numexpr as ne
import sklearn.metrics as sm

################################################################################################################################################      

################################################################################################################################################      


def doMCC(predictions,OutSampleData):
    results=dict(TP=0,TN=0,FP=0,FN=0)
    count=-1
    for X in predictions:
        count+=1
        true=OutSampleData[count,0]
        if X>0 and true==1: results['TP']+=1;continue
        elif X<0  and true==-1:results['TN']+=1;continue
        elif X>0 and true==-1: results['FP']+=1; continue
        elif X<0 and true==1: results['FN']+=1
    
    percentError=(results['FP']+results['FN'])/float(len(predictions))
    MCC=((results['TP']*results['TN'])-(results['FP']*results['FN']))/(np.sqrt((results['TP']+results['FP'])*(results['TP']+results['FN'])*(results['TN']+results['FP'])*(results['TN']+results['FN']))+.000000000000000000000000000001)

    return [MCC,percentError,results]
    
def GetErrorDual(InSampleData,OutSampleData, K,Kdecomp,a,b,giveMCC=True,givePercentError=True,giveAUC=True):
   
    H=OutSampleData[:,0].size;
    predictions=[];y_in=InSampleData[:,0].reshape(len(InSampleData[:,0]),1);y_out=OutSampleData[:,0].reshape(1,len(OutSampleData[:,0]))
    z=ne.evaluate('a*y_in')
    for i in xrange(H):
        X=b+np.dot(K[i],z)
        predictions.append(X)
   
    '''  
    import matplotlib.pyplot as pl
    data=OutSampleData
    K=Kdecomp
    pl.subplot(1,3,1)
    pl.title("Original space")
    pl.plot([data[i,1] for i in range(len(predictions)) if data[i,0]<0], [data[i,2] for i in range(len(predictions)) if data[i,0]<0], "ro")
    pl.plot([data[i,1] for i in range(len(predictions)) if data[i,0]>0], [data[i,2] for i in range(len(predictions)) if data[i,0]>0], "bo")
    pl.xlabel("x")
    pl.ylabel("y")
    
    pl.subplot(1,3,2)
    pl.title("Kernel space with guessed-predictions")
    pl.plot([K[i,0] for i in range(len(predictions)) if predictions[i]<0], [K[i,1] for i in range(len(predictions)) if predictions[i]<0], "ro")
    pl.plot([K[i,0] for i in range(len(predictions)) if predictions[i]>0], [K[i,1] for i in range(len(predictions)) if predictions[i]>0], "bo")
    pl.xlabel("first component")
    pl.ylabel("second component")
    
    pl.subplot(1,3,3)
    pl.title("Kernel space with true-predictions")
    pl.plot([K[i,0] for i in range(len(predictions)) if data[i,0]<0], [K[i,1] for i in range(len(predictions)) if data[i,0]<0], "ro")
    pl.plot([K[i,0] for i in range(len(predictions)) if data[i,0]>0], [K[i,1] for i in range(len(predictions)) if data[i,0]>0], "bo")
    pl.xlabel("first component")
    pl.ylabel("second component")
    
    pl.show()
    '''
    
    if giveMCC==True: temp=doMCC(predictions,OutSampleData); MCC=round(temp[0],2);percentError=round(temp[1],4);results=temp[2]
    if giveMCC==False: temp=ne.evaluate('y_out*predictions'); percentError=round(.5*(len(predictions)-ne.evaluate('sum(temp)'))/float(len(predictions)),4)
    if giveAUC==True: AUC=round(sm.auc_score(OutSampleData[:,0],np.array(predictions)),2)
    if giveMCC==True and giveAUC==True: return  {'MCC':MCC,'results':results,'%E':percentError,'AUC':AUC,'predictons':None}
    if giveMCC==True and giveAUC==False: return {'MCC':MCC,'results':results,'%E':percentError,'predictons':None}
    if giveMCC==False and giveAUC==False: return {'AUC':None,'%E':percentError}
################################################################################################################################################
def GetErrorPrimal(data, K,w,b,giveMCC=True,givePercentError=True,giveAUC=True):
    
    predictions=[]; ys=data[:,0].reshape(1,len(data[:,0]))
    predictions=b+np.dot(K,w)
    
    '''
    import matplotlib.pyplot as pl
    pl.subplot(1,3,1)
    pl.title("Original space")
    pl.plot([data[i,1] for i in range(len(predictions)) if data[i,0]<0], [data[i,2] for i in range(len(predictions)) if data[i,0]<0], "ro")
    pl.plot([data[i,1] for i in range(len(predictions)) if data[i,0]>0], [data[i,2] for i in range(len(predictions)) if data[i,0]>0], "bo")
    pl.xlabel("x")
    pl.ylabel("y")
    
    pl.subplot(1,3,2)
    pl.title("Kernel space predicted")
    pl.plot([K[i,0] for i in range(len(predictions)) if predictions[i]<0], [K[i,1] for i in range(len(predictions)) if predictions[i]<0], "ro")
    pl.plot([K[i,0] for i in range(len(predictions)) if predictions[i]>0], [K[i,1] for i in range(len(predictions)) if predictions[i]>0], "bo")
    pl.xlabel("first component")
    pl.ylabel("second component")
    
    pl.subplot(1,3,3)
    pl.title("Kernel space actual")
    pl.plot([K[i,0] for i in range(len(predictions)) if data[i,0]<0], [K[i,1] for i in range(len(predictions)) if data[i,0]<0], "ro")
    pl.plot([K[i,0] for i in range(len(predictions)) if data[i,0]>0], [K[i,1] for i in range(len(predictions)) if data[i,0]>0], "bo")
    pl.xlabel("first component")
    pl.ylabel("second component")
    
    pl.show()
    '''
    
    if giveMCC==True: temp=doMCC(predictions,data); MCC=round(temp[0],2);percentError=round(temp[1],4);results=temp[2]
    if giveMCC==False: temp=ne.evaluate('ys*predictions'); percentError=round(.5*(len(predictions)-ne.evaluate('sum(temp)'))/float(len(predictions)),4)
    if giveAUC==True: AUC=round(sm.auc_score(data[:,0],np.array(predictions)),2)
    if giveMCC==True and giveAUC==True: return  {'MCC':MCC,'results':results,'%E':percentError,'AUC':AUC,'predictons':None}
    if giveMCC==True and giveAUC==False: return {'MCC':MCC,'results':results,'%E':percentError,'predictons':None}
    if giveMCC==False and giveAUC==False: return {'AUC':None,'%E':percentError}