import numpy as np

def matcg(jac_vec_prod,B,maxiter=10,tol=1e-50):
    
    ab_shape = np.array(B.shape)
    ab_shape[-2]=1
    ab_shape[-1]=1
    def calc_alpha(done,r,p,Ap):
        nom = (done==False)*np.sum(r*r,axis=(-2,-1)) + 0.0
        denom = (done==False)*np.sum(p*Ap,axis=(-2,-1)) + (done==True)*1.0
        #print(denom)
        return (nom/denom).reshape(ab_shape)
    def calc_beta(done,rplus,r):
        nom = (done==False)*np.sum(rplus*rplus,axis=(-2,-1))+0.0
        denom = (done==False)*np.sum(r*r,axis=(-2,-1))+ (done==True)*1.0
        return (nom/denom).reshape(ab_shape)
    x=0.0
    done=False
    r = np.copy(B)
    p = np.copy(r)
    for k in range(0,maxiter):
        Ap = jac_vec_prod(p)

        alpha = calc_alpha(done,r,p,Ap)
        x = x + alpha*p
        rplus = r - alpha*Ap
        
        done = (np.sum(rplus**2,axis=(-2,-1))<=tol)
        

        beta = calc_beta(done,rplus,r)

        p = np.copy(rplus+beta*p)
        r = np.copy(rplus)
    return x
    

if __name__=="__main__":
    from jacobian import jac_vec_prod,get_jacobian,fn
    from scipy.sparse.linalg import cg
    np.random.seed(0)
    N=4
    Y=np.random.randn(N,N)
    Y=Y.T@Y
    
    X=np.random.randn(N,N)
    X=X.T@X
    X=np.concatenate([[np.eye(N)],[X]],0)
    Y=np.concatenate([[Y],[Y]],0)
    A=get_jacobian(X[0])
    
    
    
    Apfn = lambda x: jac_vec_prod(X,x)

    x1=matcg(Apfn,-fn(X,Y),maxiter=1)
    x12=matcg(Apfn,-fn(X,Y),maxiter=100)
    print(x1[1]-x12[1])
    
    idx=1
    A=get_jacobian(X[idx])
    b = -fn(X[idx],Y[idx]).ravel()
    x2,info =cg(A,b,maxiter=1,tol=1e-50)
    x21,info =cg(A,b,maxiter=40,tol=1e-50)
    print(x2)
    print(x21)
    print(np.linalg.norm(x2-x21))
    print(np.linalg.norm(x2.reshape(N,N)-x1[idx]))
    
    