import numpy as np

##Low rank
## (X.T@X)^2 = Y^2

def fn_lr(X,Y):
    U=X.T@X
    return U@U-Y
    
    
def trans_jac_vec_prod_lr(X,v):
    U=X.T@X
    a= U.T@v + v@U.T
    return X@a.T+X@a
    
    
def jac_vec_prod_lr(X,v):
    U=X.T@X
    dU = X.T@v+v.T@X
    return U@dU+dU@U
    

## Low rank Pertubation: (K+X.T@X)^2 = K^2+Y

def fn_lr_pert(X,K,Y):
    U=X.T@X
    A=U+K
    return A@A-Y

def trans_jac_vec_prod_lr_pert(X,K,v):
    U2=(X.T@X+K).T
    
    a= U2@v + v@U2
    return X@(a.T+a)
    
    #return (X@v.T)@U2.T+ (X@U2.T)@v.T + (X@U2)@v+(X@v)@U2
    

    
def jac_vec_prod_lr_pert(X,K,v):
    U2=X.T@X+K
    Z = X.T@v
    dU = Z+Z.T
    return U2@dU+dU@U2
    
    #return (U2@X.T)@v+(U2@v.T)@X+X.T@(v@U2)+v.T@(X@U2)

