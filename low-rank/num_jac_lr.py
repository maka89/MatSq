import numpy as np

from jacobian_lr import jac_vec_prod_lr,fn_lr,trans_jac_vec_prod_lr

def meta_fn(x,Y,M,N):
    X=x.reshape(M,N)
    F=fn_lr(X,Y)
    return F.ravel()
    
def num_jac_vec_lr(X,Y,V,eps=1e-6,trans = False):
    N=X.shape[1]
    M=X.shape[0]
    x=X.ravel()
    
    DF = np.zeros((N*N,M*N))
    for i in range(0,M*N):
        xp = np.copy(x)
        xm = np.copy(x)
        xp[i]+=eps
        xm[i]-=eps
        DF[:,i]= 0.5*(meta_fn(xp,Y,M,N)-meta_fn(xm,Y,M,N))/eps
    
    v=V.ravel()
    if trans:
        return (DF.T@v).reshape(M,N)
    else:
        return (DF@v).reshape(N,N)


    
if __name__=="__main__":
    
    ## Check reg
    M=2
    N=4
    X=np.random.randn(M,N)
    V=np.random.randn(M,N)
    Y=np.random.randn(N,N)
    print("JVP:")
    print(jac_vec_prod_lr(X,V))
    print(num_jac_vec_lr(X,Y,V))
    
    ##Check trans
    
    X=np.random.randn(M,N)
    V=np.random.randn(N,N)
    Y=np.random.randn(N,N)
    print()
    print()
    print("TRANS JVP:")
    print(trans_jac_vec_prod_lr(X,V))
    print(num_jac_vec_lr(X,Y,V,trans=True))

    M=1
    N=1000
    
    X=np.random.randn(M,N)
    V=np.random.randn(M,N)
    Y=np.random.randn(N,N)
    import time
    t0 = time.time()
    for i in range(0,100):
        y=jac_vec_prod_lr(X,V)
    print(time.time()-t0)