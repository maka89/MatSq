import numpy as np

from jacobian import jac_vec_prod,fn,trans_jac_vec_prod

def meta_fn(x,Y,N):
    X=x.reshape(N,N)
    F=fn(X,Y)
    return F.ravel()
    
def num_jac_vec(X,Y,V,eps=1e-5,trans=False):
    N=X.shape[0]
    x=X.ravel()
    
    DF = np.zeros((N*N,N*N))
    for i in range(0,N*N):
        xp = np.copy(x)
        xm = np.copy(x)
        xp[i]+=eps
        xm[i]-=eps
        
        DF[:,i]= 0.5*(meta_fn(xp,Y,N)-meta_fn(xm,Y,N))/eps
    
    v=V.ravel()
    if trans:
        return (DF.T@v).reshape(N,N)
    else:
        return (DF@v).reshape(N,N)


if __name__=="__main__":
    
    N=4
    X=np.random.randn(N,N)
    V=np.random.randn(N,N)
    Y=np.random.randn(N,N)
    
    print(jac_vec_prod(X,V))
    print(num_jac_vec(X,Y,V))
    
    print(trans_jac_vec_prod(X,V))
    print(num_jac_vec(X,Y,V,trans=True))