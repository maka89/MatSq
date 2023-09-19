import numpy as np

def matcg(jac_vec_prod,B,maxiter=10,atol=1e-50):
    
    ab_shape = np.array(B.shape)
    ab_shape[-2]=1
    ab_shape[-1]=1
    def calc_alpha(done,r,p,Ap):
        nom = (done==False)*np.sum(r*r,axis=(-2,-1)) + 0.0
        denom = (done==False)*np.sum(p*Ap,axis=(-2,-1)) + (done==True)*1.0
        return (nom/denom).reshape(ab_shape)
    def calc_beta(done,rplus,r):
        nom = (done==False)*np.sum(rplus*rplus,axis=(-2,-1))+0.0
        denom = (done==False)*np.sum(r*r,axis=(-2,-1))+ (done==True)*1.0
        return (nom/denom).reshape(ab_shape)
        
    atol_sq = atol**2
    x=np.zeros_like(B)
    r = np.copy(B)
    done = (np.sum(r**2,axis=(-2,-1))<=atol_sq)
    if np.all(done):
        return x
    p = np.copy(r)
    for k in range(0,maxiter):
        Ap = jac_vec_prod(p)

        alpha = calc_alpha(done,r,p,Ap)
        x = x + alpha*p
        rplus = r - alpha*Ap
        
        done = (np.sum(rplus**2,axis=(-2,-1))<=atol_sq)
        if np.all(done):
            print("DONE",k)
            return x

        beta = calc_beta(done,rplus,r)

        p = np.copy(rplus+beta*p)
        r = np.copy(rplus)
    return x
    

if __name__=="__main__":
    from jacobian import jac_vec_prod,get_jacobian,fn
    from scipy.sparse.linalg import cg
    np.random.seed(1)
    N=50
    Y=np.random.randn(N,N)
    Y=Y.T@Y
    
    X=np.random.randn(N,N)
    X=X.T@X
    X=np.concatenate([[np.eye(N)],[X]],0)
    Y=np.concatenate([[Y],[Y]],0)
    A=get_jacobian(X[0])
    
    
    
    Apfn = lambda x: jac_vec_prod(X,x)

    x1=matcg(Apfn,-fn(X,Y),maxiter=5000,atol=1e-9)
    
    idx=1
    A=get_jacobian(X[idx])
    b = -fn(X[idx],Y[idx]).ravel()
    x2,info =cg(A,b,maxiter=5000,atol=1e-9,tol=0.0)
    print(info)
    print(x2.reshape(N,N))
    print(x1[idx])
    print(np.linalg.norm(x2.reshape(N,N)-x1[idx]))
    
    