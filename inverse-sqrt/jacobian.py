import numpy as np
    
def fn(X,Y):
    return X@Y@X-np.eye(X.shape[0])
    
def trans_jac_vec_prod(X,Y,v):
    return Y.T@X.T@v + v@X.T@Y.T
    
def jac_vec_prod(X,Y,v):
    return X@Y@v+v@Y@X

    

def cg_step(X,B):

    alpha = np.sum(B**2)/np.sum(B*jac_vec_prod(X,B))
    return alpha*B


if __name__=="__main__":
    
    pass