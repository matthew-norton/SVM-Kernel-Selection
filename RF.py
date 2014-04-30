import matplotlib.pyplot as pl
import numpy as np
from C_SVM import *
from Kernels import *
from OptimalKernels import *

#run cross validation for kernels and C-range
def run_opC_range(dataTrain,dataTest,Klist,namelist,reduced_dimen): 
    R=[] #results=R list
    p_weight=1-(np.histogram(dataTrain[:,0], bins=2)[0][1]/float(len(dataTrain[:,0])))
    n_weight=1-p_weight
    
    a=.01*2**0
    while a<=.01*2**20:
        R.append([OptimalDualK_CSVM(a*p_weight,a*n_weight,dataTrain,dataTest,Klist,reduced_dimen=reduced_dimen),round(a,3),round(a,3)])       
        a=a*2

    return R
def run_maxC_range(dataTrain,dataTest,Klist,namelist,reduced_dimen): 
    D=max_align(dataTrain,Klist,reduced_dimen)
    Kdecomp=D['kernel']
    linearKfam=KernelFamily(dataTrain,dataTest,reduced_dimen=reduced_dimen,linear=1)[0]
    linearK=linearKfam[0]
    reduced_dimen=D['reduced_dimen']
    
    '''
    DONT FORGET TO SET REDUCED_DIMEN==DEFAULT
    Kdecomp_sum=Klist[0]
    for i in range(1,len(Klist)): Kdecomp_sum+=Klist[i]
    print Kdecomp_sum.shape
    
    from sklearn.preprocessing import KernelCenterer
    K=np.dot(Kdecomp_sum,Kdecomp_sum.T)
    Kcent=KernelCenterer()
    K=Kcent.fit_transform(K)
    
    K=K/float(np.trace(K))
    [EigVals,EigVectors]=scipy.sparse.linalg.eigsh(K,k=reduced_dimen,which='LM')
    
    #[EigVals,EigVectors]=scipy.linalg.eigh(K,eigvals=(len(K[:,0])-reduced_dimen,len(K[:,0])-1))
    EigVals=np.flipud(np.fliplr(np.diag(EigVals)))
    
    EigVectors=np.fliplr(EigVectors)
    Kdecomp=np.dot(EigVectors,scipy.linalg.sqrtm(EigVals))
    Kdecomp=normalize(Kdecomp,copy=False)
    print Kdecomp.shape
    '''
    pl.figure(figsize=(15,8))
    
    pl.subplot(2,2,1)
    pl.title("PCA training space")
    pl.plot([linearK[i,0] for i in range(len(dataTrain[:,0])) if dataTrain[i,0]<0], [linearK[i,1] for i in range(len(dataTrain[:,0])) if dataTrain[i,0]<0], "ro")
    pl.plot([linearK[i,0] for i in range(len(dataTrain[:,0])) if dataTrain[i,0]>0], [linearK[i,1] for i in range(len(dataTrain[:,0])) if dataTrain[i,0]>0], "bo")
    pl.xlabel("x")
    pl.ylabel("y")
    
    pl.subplot(2,2,2)
    pl.title("PCA testing space")
    pl.plot([linearK[i,0] for i in range(len(dataTest[:,0])) if dataTest[i,0]<0], [linearK[i,1] for i in range(len(dataTest[:,0])) if dataTest[i,0]<0], "ro")
    pl.plot([linearK[i,0] for i in range(len(dataTest[:,0])) if dataTest[i,0]>0], [linearK[i,1] for i in range(len(dataTest[:,0])) if dataTest[i,0]>0], "bo")
    pl.xlabel("x")
    pl.ylabel("y")
    
    pl.subplot(2,2,3)
    pl.title("Kernel PCA train space")
    pl.plot([Kdecomp[i,0] for i in range(len(dataTrain[:,0])) if dataTrain[i,0]<0], [Kdecomp[i,1] for i in range(len(dataTrain[:,0])) if dataTrain[i,0]<0], "ro")
    pl.plot([Kdecomp[i,0] for i in range(len(dataTrain[:,0])) if dataTrain[i,0]>0], [Kdecomp[i,1] for i in range(len(dataTrain[:,0])) if dataTrain[i,0]>0], "bo")
    pl.xlabel("first component")
    pl.ylabel("second component")
    
    pl.subplot(2,2,4)
    pl.title("Kernel PCA test space")
    pl.plot([Kdecomp[i,0] for i in range(len(dataTest[:,0])) if dataTest[i,0]<0], [Kdecomp[i,1] for i in range(len(dataTest[:,0])) if dataTest[i,0]<0], "ro")
    pl.plot([Kdecomp[i,0] for i in range(len(dataTest[:,0])) if dataTest[i,0]>0], [Kdecomp[i,1] for i in range(len(dataTest[:,0])) if dataTest[i,0]>0], "bo")
    pl.xlabel("first component")
    pl.ylabel("second component")
    for i in range(len(Klist)): print D['u'][i],namelist[i]
    pl.show()
    
    
    print PrimalK_CSVM(1,1,dataTrain,dataTest,Kdecomp,reduced_dimen)
    
    R=[] #results=R list
    p_weight=1-(np.histogram(dataTrain[:,0], bins=2)[0][1]/float(len(dataTrain[:,0])))
    n_weight=1-p_weight
    
    import time
    a=.01*2**0
    while a<=.01*2**20:
        s=time.time()
        R.append([PrimalK_CSVM(a*p_weight,a*n_weight,dataTrain,dataTest,Kdecomp,reduced_dimen),round(a,3),round(a,3)])        
        f=time.time()
        print 'finish',f-s
        print a, 'a'*50
        a=a*2
        
    return R 
def run_regC_range(dataTrain,dataTest,Klist,namelist,reduced_dimen): 
    K=Klist[0]
    R=[] #results=R list
    p_weight=1-(np.histogram(dataTrain[:,0], bins=2)[0][1]/len(dataTrain[:,0]))
    n_weight=1-p_weight
    
    a=.01*2**10
    while  not_stop and a<=.01*2**10:
        R.append([PrimalK_CSVM(a*p_weight,a*n_weight,dataTrain,dataTest,K,reduced_dimen=reduced_dimen),round(a,3),round(a,3)])
        a=a*1.5
        
        
    return R   



#!!steps!!
#get data
from getdata import *
data=get_digits(digit=0)
dataTrain=data[0]#np.concatenate((data[0],data[1]))
dataTest=data[1]


#choose reduced dimention, get kernel matrices to try
reduced_dimen=10

#choose error measure
errorMeasure='MCC'
opC=False
maxC=True
regC=False

#choose num_k_folds
num_folds=2

#get Kernels
from Kernels import *
linear=0
gaussian=[2**x for x in range(-5,5)]
tstudent=[2**x for x in range(-5,5) ]
polynomial=[x for x in range(1,10)]
cauchy=[x for x in [-.1,-.05,.05,.1,1,3,5,7,10,13,16]]
power=[x for x in [.001,.01,.5,5,10,15,20,40,60,80,100]]
wave=[x for x in range(0)]
sigmoid=[x for x in [-.0001,-.001,-.01,-.1,-.5,-.9,-1.5,-2]]

Kfam=KernelFamily(dataTrain,dataTest,reduced_dimen=reduced_dimen,tstudent=[2**x for x in range(-5,10)],gaussian=[2**x for x in range(-5,10) ],polynomial=[x for x in range(1,9)])
Klist=Kfam[0]
namelist=Kfam[1]
#RUN IT!!!
R=[[] for x in range(num_folds)]

if num_folds==1:
    if regC: R[0].append(run_regC_range(dataTrain,dataTest,Klist,namelist,reduced_dimen))
    if opC: R[0].append(run_opC_range(dataTrain,dataTest,Klist,namelist,reduced_dimen))
    if maxC: R[0].append(run_maxC_range(dataTrain,dataTest,Klist,namelist,reduced_dimen))
if num_folds>1:
    fold_size=len(dataTrain[:,0])//num_folds
    data_fold=[ [] for x in range(num_folds) ]
    Klist_folds=[ [] for x in range(len(Klist)) ]
# get folds
    data_fold=[ dataTrain[i*fold_size:(i+1)*fold_size,:]  for i in range(num_folds) ]
    for i in range(len(Klist)): 
        Klist_folds[i]=[Klist[i][j*fold_size:(j+1)*fold_size,:] for j in range(num_folds)]
#build proper data and K from folds for fold #i
    for i in range(num_folds):
        dataTrain=np.vstack((data_fold[j] for j in range(num_folds) if j!=i))
        dataTest=data_fold[i] #leaving out fold i
        Knew_list=[x for x in range(len(Klist))]
        for x in range(len(Klist)):
            Knew_list[x]=np.vstack((Klist_folds[x][j] for j in range(num_folds) if j!=i))
            Knew_list[x]=np.vstack((Knew_list[x],Klist_folds[x][i]))
        #run fold
        if regC: R[i].append(run_regC_range(dataTrain,dataTest,Knew_list,namelist,reduced_dimen))
        if opC: R[i].append(run_opC_range(dataTrain,dataTest,Knew_list,namelist,reduced_dimen))
        if maxC: R[i].append(run_maxC_range(dataTrain,dataTest,Knew_list,namelist,reduced_dimen))
        
#note R[i][0]=list of [OptimalDualK_CSVM(a,a,dataTrain,dataTest,Klist,reduced_dimen=reduced_dimen),round(a,3),round(a,3)] for every  C-value tested for fold # i-1
#note R[i][0][j]= [OptimalDualK_CSVM(a,a,dataTrain,dataTest,Klist,reduced_dimen=reduced_dimen),round(a,3),round(a,3)] for the j'th C-value tested
#note R[i][0][j][0]= dictionary =OptimalDualK_CSVM(a,a,dataTrain,dataTest,Klist,reduced_dimen=reduced_dimen)
#note R[i][0][j][1]=j'th C-value tested= round(a,3)   OR j'th (positive_penalty, negative_penalty) tested=(round(a,3),round(b,3))
#note R[i][0][j][0] contains dict=R[i][j][0]['in'] and dict=R[i][j][0]['out']




#get 'good' folds
avgSumDiff_for_fold=[]
varSumDiff_for_fold=[]
in_out_squared_Diffs_for_fold=[]
for j in range(num_folds):
    if sum(R[j][0][i][0]['out'][errorMeasure] for i in range(len(R[0][0])))==0:
        in_out_squared_Diffs_for_fold.append(10**100)
        avgSumDiff_for_fold.append(10**100)
        varSumDiff_for_fold.append(10**100)
    else:
        in_out_squared_Diffs_for_fold.append([max(R[j][0][i][0]['in'][errorMeasure],0)-max(R[j][0][i][0]['out'][errorMeasure],0)**2 for i in range(len(R[0][0]))])
        avgSumDiff_for_fold.append(np.mean(in_out_squared_Diffs_for_fold[j]))
        varSumDiff_for_fold.append(np.var(in_out_squared_Diffs_for_fold[j]))
    print avgSumDiff_for_fold[j],varSumDiff_for_fold[j], 'fold '+str(j)
    
folds_to_use=[]
for j in range(num_folds):
    if avgSumDiff_for_fold[j]<np.percentile(avgSumDiff_for_fold,70) and varSumDiff_for_fold[j]>np.percentile(varSumDiff_for_fold,0) : 
        folds_to_use.append(j)
print folds_to_use

#plot errors for all folds
pl.figure(figsize=(15,9))
for i in range(num_folds):
    pl.subplot(2,(num_folds+1)//2,i+1,yscale=('log'))
    if i in folds_to_use: pl.title('GOOD fold #'+str(i))
    else: pl.title('BAD fold #'+str(i))
    in_plot,=pl.plot([R[i][0][x][0]['in'][errorMeasure] for x in range(len(R[i][0]))],[R[i][0][x][1] for x in range(len(R[i][0]))], "ro")
    out_plot,=pl.plot([R[i][0][x][0]['out'][errorMeasure] for x in range(len(R[i][0]))],[R[i][0][x][1] for x in range(len(R[i][0]))],  "bo")
    pl.xlabel("Error Measure = " +errorMeasure)
    pl.ylabel("C-value")
    print '\n', 'Fold'+str(i)
    for j in R[i][0]: print j
pl.show() 

      
#plot avg fold error for C-values

pl.figure(figsize=(15,9)) 

#plot for good fold rules
pl.subplot(2,1,1,yscale=('log'))   

avgE_in=[ [] for x in range(len(R[0][0]))]
avgE_out=[ [] for x in range(len(R[0][0]))]
stdE_in=[ [] for x in range(len(R[0][0]))]
stdE_out=[ [] for x in range(len(R[0][0]))]
cvar_in=[ [] for x in range(len(R[0][0]))]
cvar_out=[ [] for x in range(len(R[0][0]))]
for i in range(len(R[0][0])):
    #cvar_in[i]=sum(np.percentile([max(R[j][0][i][0]['in'][errorMeasure],0) for j in folds_to_use],[x*10 for x in range(3,9)]))/float(6)
    #cvar_out[i]=sum(np.percentile([max(R[j][0][i][0]['out'][errorMeasure],0) for j in folds_to_use],[x*10 for x in range(3,9)]))/float(6)

    stdE_in[i]=np.std([max(R[j][0][i][0]['in'][errorMeasure],0) for j in folds_to_use])
    stdE_out[i]=np.std([max(R[j][0][i][0]['out'][errorMeasure],0) for j in folds_to_use])
    avgE_in[i]=np.mean([max(R[j][0][i][0]['in'][errorMeasure],0) for j in folds_to_use]  )
    avgE_out[i]=np.mean([max(R[j][0][i][0]['out'][errorMeasure],0) for j in folds_to_use])
#plot CVaR
'''
pl.plot(cvar_in,[R[0][0][x][1] for x in range(len(R[0][0]))], "r-")
pl.plot(cvar_out,[R[0][0][x][1] for x in range(len(R[0][0]))], "b-")
'''
#plot standard deviation
pl.plot([avgE_in[i]-stdE_in[i] for i in range(len(R[0][0]))],[R[0][0][x][1] for x in range(len(R[0][0]))], "k+")
pl.plot([avgE_in[i]+stdE_in[i] for i in range(len(R[0][0]))],[R[0][0][x][1] for x in range(len(R[0][0]))], "k+")
pl.plot([avgE_out[i]-stdE_out[i] for i in range(len(R[0][0]))],[R[0][0][x][1] for x in range(len(R[0][0]))], "m+")
pl.plot([avgE_out[i]+stdE_out[i] for i in range(len(R[0][0]))],[R[0][0][x][1] for x in range(len(R[0][0]))], "m+")
#plot avg error
pl.plot(avgE_in,[R[0][0][x][1] for x in range(len(R[0][0]))], "ro")
pl.plot(avgE_out,[R[0][0][x][1] for x in range(len(R[0][0]))],  "bo-")
pl.xlabel('avg Error over '+str(len(folds_to_use))+' GOOD folds')
pl.ylabel("C-value")


#plot for all folds
pl.subplot(2,1,2,yscale=('log'))

avgE_in=[ [] for x in range(len(R[0][0]))]
avgE_out=[ [] for x in range(len(R[0][0]))]
stdE_in=[ [] for x in range(len(R[0][0]))]
stdE_out=[ [] for x in range(len(R[0][0]))]
cvar_in=[ [] for x in range(len(R[0][0]))]
cvar_out=[ [] for x in range(len(R[0][0]))]
for i in range(len(R[0][0])):
    #cvar_in[i]=sum(np.percentile([max(R[j][0][i][0]['in'][errorMeasure],0) for j in range(num_folds)],[x*10 for x in range(3,9)]))/float(6)
    #cvar_out[i]=sum(np.percentile([max(R[j][0][i][0]['out'][errorMeasure],0) for j in range(num_folds)],[x*10 for x in range(3,9)]))/float(6)

    stdE_in[i]=np.std([max(R[j][0][i][0]['in'][errorMeasure],0) for j in range(num_folds)])
    stdE_out[i]=np.std([max(R[j][0][i][0]['out'][errorMeasure],0) for j in range(num_folds)])
    avgE_in[i]=np.mean([max(R[j][0][i][0]['in'][errorMeasure],0) for j in range(num_folds)]  )
    avgE_out[i]=np.mean([max(R[j][0][i][0]['out'][errorMeasure],0) for j in range(num_folds)])
#plot CVaR
'''
pl.plot(cvar_in,[R[0][0][x][1] for x in range(len(R[0][0]))], "r-")
pl.plot(cvar_out,[R[0][0][x][1] for x in range(len(R[0][0]))], "b-")
'''
#plot standard deviation
pl.plot([avgE_in[i]-stdE_in[i] for i in range(len(R[0][0]))],[R[0][0][x][1] for x in range(len(R[0][0]))], "k+")
pl.plot([avgE_in[i]+stdE_in[i] for i in range(len(R[0][0]))],[R[0][0][x][1] for x in range(len(R[0][0]))], "k+")
pl.plot([avgE_out[i]-stdE_out[i] for i in range(len(R[0][0]))],[R[0][0][x][1] for x in range(len(R[0][0]))], "m+")
pl.plot([avgE_out[i]+stdE_out[i] for i in range(len(R[0][0]))],[R[0][0][x][1] for x in range(len(R[0][0]))], "m+")
#plot avg error
pl.plot(avgE_in,[R[0][0][x][1] for x in range(len(R[0][0]))], "ro")
pl.plot(avgE_out,[R[0][0][x][1] for x in range(len(R[0][0]))],  "bo-")
pl.xlabel('avg Error over '+str(num_folds)+'total folds')
pl.ylabel("C-value")


pl.show()
