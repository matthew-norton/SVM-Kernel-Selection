import numpy as np
import numexpr as ne
import gurobipy
from gurobipy import *
from GetResults import *

################################################################################################################################################      

################################################################################################################################################      

def PrimalK_CSVM(positive_penalty,negative_penalty,InSampleData,OutSampleData,K,reduced_dimen):

#initiate problem
    
    model=Model("PrimalK_CSVM")
    in_samples=InSampleData[:,0].size
    out_samples=OutSampleData[:,0].size

#add Kernel Matrix and G=y *K* yT
# K = V * sqrt(Lambda)
    y_rowform=InSampleData[:,0].reshape(1,in_samples)
    y_colform=y_rowform.reshape(in_samples,1)

#add variables

    w=[]
    E=[]
    
    for i in xrange(reduced_dimen):
        w.append(model.addVar(lb=-10**21,vtype=GRB.CONTINUOUS))
    for i in xrange(in_samples):
        E.append(model.addVar(lb=0.0,vtype=GRB.CONTINUOUS))
    b=model.addVar(lb=-10**21)
    
#set objective 

    expr1=QuadExpr()
    for i in xrange(reduced_dimen):
        expr1.addTerms(.5,w[i],w[i])
    expr2=LinExpr([positive_penalty for i in xrange(in_samples) if y_colform[i]>0],[E[i] for i in xrange(in_samples) if y_colform[i]>0])
    expr3=LinExpr([negative_penalty for i in xrange(in_samples) if y_colform[i]<0],[E[i] for i in xrange(in_samples) if y_colform[i]<0])
    model.update()
    model.setObjective( expr1+expr2+expr3 ,GRB.MINIMIZE)
    
#add contraints

    for i in xrange(in_samples):
        model.addConstr(y_rowform[0,i]*(b+quicksum(  w[j]*K[i,j] for j in xrange(reduced_dimen))  )>=1-E[i])
    
#get solution
 
    model.setParam('PSDTol',1000000000000000000)
    model.optimize()
    w=model.getVars()[0:reduced_dimen]
    b=b.X
    for i in range(reduced_dimen): w[i]=w[i].X
    w=np.array(w,dtype='float64').reshape(reduced_dimen,1)

#return error as dictionary
    
    return {'in':GetErrorPrimal(InSampleData,K[0:in_samples],w,b,giveMCC=True,giveAUC=True),'out':GetErrorPrimal(OutSampleData,K[in_samples:],w,b,giveMCC=True,giveAUC=True)}
    

################################################################################################################################################      



################################################################################################################################################      

def DualK_CSVM(positive_penalty,negative_penalty,InSampleData,OutSampleData,Kdecomp,reduced_dimen):

#initiate problem
 
    print 'in problem'
    model=Model("qp")
    K=np.dot(Kdecomp,Kdecomp.T)
    in_samples=InSampleData[:,0].size

#add Kernel Matrix and G=y *K* yT

    KTrain=K[     0:in_samples   ,   0:in_samples    ]
    y_rowform=InSampleData[:,0].reshape(1,in_samples)
    y_colform=y_rowform.reshape(in_samples,1)
    G=ne.evaluate('KTrain*y_rowform*y_colform')

#add variables

    a=[]
    for i in xrange(in_samples):
        if y_colform[i]>0: a.append(model.addVar(lb=0.0,ub=positive_penalty,vtype=GRB.CONTINUOUS))
        elif y_colform[i]<0: a.append(model.addVar(lb=0.0,ub=negative_penalty,vtype=GRB.CONTINUOUS))
    model.update()

#set objective 

    expr1=LinExpr()
    for var in a: expr1.addTerms(2.0,var)
    expr2=QuadExpr()
    for i in xrange(in_samples):
        for j in xrange(in_samples):
            expr2.addTerms(G[i,j],a[i],a[j])
    
    model.setObjective( expr1-expr2 ,GRB.MAXIMIZE)

    
#add contraints

    model.addConstr(quicksum(a[i]*y_rowform[0,i] for i in xrange(in_samples))==0)
    
#get solution

    model.setParam('PSDTol',1000000000000000000)
    model.optimize()
    a=model.getVars()
    for i in range(len(a)): a[i]=a[i].X
    a=np.array(a,dtype='float64').reshape(in_samples,1)
    
#solve for b-intercept

    Mindex=None
    for i in xrange(in_samples):
        if y_colform[i]>0 and round(a[i],7)>0 and round(a[i],7)<positive_penalty:  M=a[i];  Mindex=i;  break
        elif y_colform[i]<0 and round(a[i],7)>0 and round(a[i],7)<negative_penalty: M=a[i];  Mindex=i; break
    if Mindex==None: Mindex=0;print 'error: check b value'
    K_Mindex_col=KTrain[:,Mindex].reshape(in_samples,1)
    temp=ne.evaluate('a*y_colform*K_Mindex_col')
    b=y_colform[Mindex]-ne.evaluate('sum(temp)')#sum([a[i]*y[i]*KTrain[i,Mindex] for i in xrange(in_samples)])
    
#return error as dictionary

    KTest= K[     in_samples :   ,   0:in_samples    ]
    return {'in':GetErrorDual(InSampleData,InSampleData,KTrain,Kdecomp[0:in_samples],a,b,giveMCC=True,giveAUC=False),'out':GetErrorDual(InSampleData,OutSampleData,KTest,Kdecomp[in_samples:],a,b,giveMCC=True,giveAUC=False)}
################################################################################################################################################      