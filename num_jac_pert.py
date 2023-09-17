from jacobian_lr import jac_vec_prod_lr_pert,fn_lr_pert,trans_jac_vec_prod_lr_pert
import numpy as np
def meta_fn(x,K,Y,M,N):
    X=x.reshape(M,N)
    F=fn_lr_pert(X,K,Y)
    return F.ravel()
    
def num_jac_vec_lr_pert(X,K,Y,V,eps=1e-6,trans = False):
    N=X.shape[1]
    M=X.shape[0]
    x=X.ravel()
    
    DF = np.zeros((N*N,M*N))
    for i in range(0,M*N):
        xp = np.copy(x)
        xm = np.copy(x)
        xp[i]+=eps
        xm[i]-=eps
        DF[:,i]= 0.5*(meta_fn(xp,K,Y,M,N)-meta_fn(xm,K,Y,M,N))/eps
    
    v=V.ravel()
    if trans:
        return (DF.T@v).reshape(M,N)
    else:
        return (DF@v).reshape(N,N)

if __name__=="__main__":
     ## Check reg
    M=20
    N=40
    X=np.random.randn(M,N)
    V=np.random.randn(M,N)
    Y=np.random.randn(N,N)
    
    K=np.random.randn(N,N)

    x1=jac_vec_prod_lr_pert(X,K,V)
    x2=num_jac_vec_lr_pert(X,K,Y,V)


    print("JVP:",np.sqrt(np.linalg.norm(x1-x2)),np.sqrt(np.linalg.norm(x1)))
        
    ##Check trans
    
    X=np.random.randn(M,N)
    V=np.random.randn(N,N)
    Y=np.random.randn(N,N)
    print()
    print()
    
    x1=trans_jac_vec_prod_lr_pert(X,K,V)
    x2=num_jac_vec_lr_pert(X,K,Y,V,trans=True)
    print("TRANS JVP:",np.sqrt(np.linalg.norm(x1-x2)),np.sqrt(np.linalg.norm(x1)))
    
    
    M=1
    N=1000
    
    X=np.random.randn(M,N)
    V=np.random.randn(M,N)
    Y=np.random.randn(N,N)
    K=np.random.randn(N,N)
    import time
    t0 = time.time()
    for i in range(0,100):
        #y=trans_jac_vec_prod_lr_pert(X,K,V)
        y=jac_vec_prod_lr_pert(X,K,V)
    print(time.time()-t0)