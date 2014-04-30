import numpy as np
import numexpr as ne
import scipy
from scipy.linalg import *
import picos as pic
import gurobipy
from gurobipy import *
from GetResults import *
from sklearn.preprocessing import normalize

################################################################################################################################################      

################################################################################################################################################      

def OptimalDualK_CSVM(positive_penalty,negative_penalty,InSampleData,OutSampleData,Klist,reduced_dimen): 
#initiate problem

    model=Model("op_kernel_qp")
    in_samples=InSampleData[:,0].size
    
#add Kernel Matrix and G=y^T *K* y

    y_rowform=InSampleData[:,0].reshape(1,in_samples)
    y_colform=y_rowform.reshape(in_samples,1)
    
#add variables

    a=[]
    for i in xrange(in_samples):
        if y_colform[i]>0: a.append(model.addVar(lb=0.0,ub=positive_penalty,vtype=GRB.CONTINUOUS))
        elif y_colform[i]<0: a.append(model.addVar(lb=0.0,ub=negative_penalty,vtype=GRB.CONTINUOUS))
    t=model.addVar(lb=-10**21,vtype=GRB.CONTINUOUS)
    model.update()
    
#set objective

    expr1=LinExpr()
    for var in a: expr1.addTerms(2.0,var)
    model.setObjective( expr1-t ,GRB.MAXIMIZE)

#add contraints

    Clist=[]
    expr2=[]
    for x in range(len(Klist)):
        expr2.append(QuadExpr())
        for i in xrange(in_samples):
            for j in xrange(in_samples):
                temp=y_colform[i]*y_colform[j]*np.dot(Klist[x][i],Klist[x][j].T)
                expr2[x].addTerms(temp,a[i],a[j])
        K=Klist[x]
        temp=ne.evaluate('K*K')
        trace=ne.evaluate('sum(temp)')
        Clist.append(model.addQConstr(  t >=  expr2[x]/ trace))                         #quicksum(quicksum(Klist[x][i,j]*a[i]*y_colform[i] for i in range(in_samples) )*quicksum(Klist[x][i,j]*a[i]*y_colform[i] for i in range(in_samples) ) for j in range(reduced_dimen) )  ))#/np.dot(K[0:in_samples].flatten('F'),K[0:in_samples].flatten('F')  )))     
    model.addConstr(quicksum(a[i]*y_colform[i] for i in xrange(in_samples))==0)

#display solution

    model.setParam('PSDTol',10**100)
    model.setParam('QCPDual',1)

    model.optimize() 

    a=model.getVars()[0:in_samples]
    for i in range(len(a)): a[i]=a[i].X
    a=np.array(a,dtype='float64').reshape(in_samples,1)
    
    #these steps get your K-matrix and u-coefficients to get your new K-decomposed matrix(Not necessarily eigen decomp of optimal K)
    u=[]
    K=None
    for i in xrange(len(Klist)):
        u.append(-Clist[i].QCPi)
        if u[i]>10**-10 and K==None:
            K=np.sqrt(u[i])*Klist[i]
            print 'not none'
        elif u[i]>10**-10:
            K=np.concatenate( (K,np.sqrt(u[i])*Klist[i]),axis=1)
    w=np.dot(K[0:in_samples].T,a*y_colform)
    #w=w/np.linalg.norm(w)
    
#solve for b

    Mindex=[]
    for i in xrange(in_samples):
        print a[i]
        if y_colform[i]>0 and round(a[i],7)>0 and round(a[i],7)<positive_penalty:   Mindex=i;break
        elif y_colform[i]<0 and round(a[i],7)>0 and round(a[i],7)<negative_penalty:   Mindex=i;break
    if Mindex==[]: print 'error: check b value'*100;Mindex=0
    
    b=y_colform[Mindex]-np.dot(K[Mindex],w)
    print b,'bbbbbb',w,'wwwwwww'

#return error as dictionary

    return {'in':GetErrorPrimal(InSampleData,K[0:in_samples],w,b,giveMCC=True,giveAUC=True),'out':GetErrorPrimal(OutSampleData,K[in_samples:],w,b,giveMCC=True,giveAUC=True)}
################################################################################################################################################      




################################################################################################################################################      

def max_align(InSampleData,Klist,reduced_dimen):
    
    model=Model('max_align')
    in_samples=InSampleData[:,0].size    ; num_kernels=len(Klist)
    
#variables

    v=[]
    for i in range(num_kernels):
        v.append(model.addVar(lb=0))
    '''B=prob.add_variable('B',num_kernels,vtype='binary')'''
    
#constants

    y_rowform=InSampleData[:,0].reshape(1,in_samples)
    y_colform=y_rowform.reshape(in_samples,1)
    a=[]
    for K in Klist:
        temp=np.dot(y_colform.T,K[0:in_samples])
        a.append(np.dot(    temp   ,   temp.T   )  )
    a=np.array(a).reshape(num_kernels,1)   ; M=np.zeros((num_kernels,num_kernels),'d')
    for i in xrange(num_kernels):
        for j in xrange(num_kernels):
            K1=np.dot(Klist[i][0:in_samples].T,Klist[j][0:in_samples])
            temp=ne.evaluate('K1*K1')            #np.trace(  np.dot(   np.dot(Klist[j][0:in_samples].T, Klist[i][0:in_samples])   ,    np.dot(Klist[i][0:in_samples].T,Klist[j][0:in_samples])    )    )
            M[i,j]=ne.evaluate('sum(temp)')      #temp
    model.update()
    
#objective

    objective=quicksum(M[i,j]*v[i]*v[j] for i in xrange(num_kernels) for j in xrange(num_kernels))-2*quicksum(v[i]*a[i] for i in xrange(num_kernels))
    model.setObjective(objective,GRB.MINIMIZE)
    '''for i in range(num_kernels):
        prob.add_constraint(v[i]<B[i])
    prob.add_constraint(sum(B)<=num_kernels)'''
    
#solve

    model.setParam('PSDTol',10**100)
    model.optimize()
    
#solve for K and return solution

    v=model.getVars()
    for i in range(len(v)): v[i]=v[i].X
    v=np.array(v,dtype='float64').reshape(num_kernels,1)
    vnorm=np.linalg.norm(v);u=v/vnorm
    K=None
    count=0
    print u
    for i in xrange(len(Klist)):
        if u[i]>1*10**-6 and K==None:
            count=1
            K=np.sqrt(u[i])*Klist[i]
        elif u[i]>1*10**-6:
            count+=1
            K=np.concatenate( (K,np.sqrt(u[i])*Klist[i]),axis=1)
    
    K=np.dot(K,K.T)

    K=K/float(np.trace(K))
    #[EigVals,EigVectors]=scipy.sparse.linalg.eigsh(K,k=reduced_dimen,which='LM')
    
    [EigVals,EigVectors]=scipy.linalg.eigh(K,eigvals=(len(K[:,0])-reduced_dimen,len(K[:,0])-1))
    EigVals=np.flipud(np.fliplr(np.diag(EigVals)))
    
    EigVectors=np.fliplr(EigVectors)
    K=np.dot(EigVectors,scipy.linalg.sqrtm(EigVals))
    count=1
    
    K=normalize(K,copy=False)
    return {'kernel':K,'u':u,'reduced_dimen':count*reduced_dimen}
