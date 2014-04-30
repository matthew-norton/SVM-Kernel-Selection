import numpy as np 
import numexpr as ne
import scipy.linalg
import scipy.sparse.linalg
from sklearn.preprocessing import KernelCenterer,normalize
import sklearn.metrics as sm

################################################################################################################################################      

################################################################################################################################################   
   
def KernelFamily(InSampleData,OutSampleData,reduced_dimen,linear=0,polynomial=[],tstudent=[],gaussian=[],cauchy=[],sigmoid=[],wave=[],power=[]):
    Klist=[]
    namelist=[]
    
    
    in_samples=InSampleData.shape[0]
    no_y_data=np.vstack((InSampleData[:,1:],OutSampleData[:,1:]))
    num_samples=no_y_data.shape[0]
    Y=sm.pairwise.pairwise_distances(no_y_data,no_y_data, metric='euclidean',n_jobs=2)

    
    def center_normTrace_decomp(K):
        print 'centering kernel'
        #### Get transformed features for K_train that DONT snoop when centering, tracing, or eiging#####
        Kcent=KernelCenterer()
        Ktrain=Kcent.fit_transform(K[:in_samples,:in_samples])
        #Ktrain=Ktrain/float(np.trace(Ktrain))
        #[EigVals,EigVectors]=scipy.sparse.linalg.eigsh(Ktrain,k=reduced_dimen,which='LM')
        [EigVals,EigVectors]=scipy.linalg.eigh(Ktrain,eigvals=(in_samples-reduced_dimen,in_samples-1))
        for i in range(len(EigVals)): 
            if EigVals[i]<=0: EigVals[i]=0
        EigVals=np.flipud(np.fliplr(np.diag(EigVals)))
        EigVectors=np.fliplr(EigVectors)
        Ktrain_decomp=np.dot(EigVectors,scipy.linalg.sqrtm(EigVals))
       
        #### Get transformed features for K_test using K_train implied mapping ####
        Kcent=KernelCenterer()
        Kfull=Kcent.fit_transform(K)
        #Kfull=Kfull/float(np.trace(Kfull))
        K_train_test=Kfull[in_samples:,:in_samples]
        Ktest_decomp=np.dot(K_train_test,np.linalg.pinv(Ktrain_decomp.T))

        ####combine mapped train and test vectors and normalize each vector####
        Kdecomp=np.vstack((Ktrain_decomp,Ktest_decomp))
        print 'doing normalization'
        Kdecomp=normalize(Kdecomp,copy=False)
        return Kdecomp
        
        
    
        
                
    if linear!=0 or len(polynomial)!=0 or len(sigmoid)!=0:
        reg_dot_prod_matrix=np.inner(no_y_data,no_y_data)
        
#LINEAR
    if linear!=0:
        print 'linear'
        K=reg_dot_prod_matrix
        K=center_normTrace_decomp(K)        
        Klist.append(K)
        namelist.append('linear')
        
#POLYNOMIAL
    for i in polynomial:
        print 'polynomial'
        K=reg_dot_prod_matrix
        K=ne.evaluate('K**i')
        K=center_normTrace_decomp(K)        
        Klist.append(K)
        namelist.append('poly'+str(i))

#TSTUDENT
    for i in tstudent:
        print 'tstudent'
        K=ne.evaluate('1/(1+Y**i)',truediv=True)
        K=center_normTrace_decomp(K)        
        Klist.append(K)
        namelist.append('tstudent'+str(i))
        
#GAUSSIAN
    for i in gaussian:
        print 'gaussian'
        K=ne.evaluate('Y**2')
        K=ne.evaluate('exp(-i*K)')
        K=center_normTrace_decomp(K)
        Klist.append(K)
        namelist.append('gaussian'+str(i))
        
#CAUCHY
    for i in cauchy:
        print 'cauchy'
        K=ne.evaluate('Y**2')
        K=ne.evaluate('1/(1+i*(K))',truediv=True)
        K=center_normTrace_decomp(K)        
        Klist.append(K)
        namelist.append('cauchy'+str(i))
        
#SIGMOID
    for a in sigmoid:#A common value for alpha is 1/N, where N is the data dimension
        try:
            print 'sigmoid'
            b=1
            K=reg_dot_prod_matrix
            K=ne.evaluate('tanh(a*K+b)')
            print K
            K=center_normTrace_decomp(K)        
            Klist.append(K)
            namelist.append('sigmoid'+str(a))
        except: print 'no sigmoid with param '+str(a)
        
#WAVE
    for i in wave:
        print 'wave'
        K=ne.evaluate('(i/Y)*sin(Y/i)')
        for i in xrange(len(K[0])):K[i,i]=0
        K=center_normTrace_decomp(K)        
        Klist.append(K)
        namelist.append('wave'+str(i))
        
#POWER
    for i in power:
        print 'power'
        K=ne.evaluate('-(Y**i)')
        K=center_normTrace_decomp(K)        
        Klist.append(K)
        namelist.append('power'+str(i))

#RESULTS
    
    return [Klist,namelist]
