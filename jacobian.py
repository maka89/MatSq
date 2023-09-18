import numpy as np
from scipy.sparse.linalg import LinearOperator
    
def fn(X,Y):
    return X@X-Y
    
def trans_jac_vec_prod(X,v):
    return X.T@v + v@X.T
    
def jac_vec_prod(X,v):
    return X@v+v@X


def get_jacobian(X):
    N=X.shape[0]
    assert(X.shape[0]==X.shape[1])

    
    matvecf = lambda v:jac_vec_prod(X,v.reshape(N,N)).ravel()
    
    return LinearOperator((N*N,N*N),matvec=matvecf)
    

def cg_step(X,B):

    alpha = np.sum(B**2)/np.sum(B*jac_vec_prod(X,B))
    return alpha*B
    

if __name__=="__main__":
    
    pass