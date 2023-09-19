import torch

def calc_alpha(done,r,p,Ap):
        nom = (done==False)*torch.sum(r*r,dim=(-2,-1),keepdims=True)# + 0.0
        denom = (done==False)*torch.sum(p*Ap,dim=(-2,-1),keepdims=True) + (done==True)*1.0
        return (nom/denom)
def calc_beta(done,rplus,r):
    nom = (done==False)*torch.sum(rplus*rplus,dim=(-2,-1),keepdims=True)#+0.0
    denom = (done==False)*torch.sum(r*r,dim=(-2,-1),keepdims=True)+ (done==True)*1.0
    return (nom/denom)

def calc_done(r,tol):
    return torch.sum(r**2,dim=(-2,-1),keepdims=True)<=tol
    
def matcg(jac_vec_prod,B,maxiter=10,atol=1e-16):
      
    atol_sq = atol**2
    x=torch.zeros_like(B)
    r = torch.clone(B)
    
    done = calc_done(r,atol_sq)
    if torch.all(done):
        return x
        
    p = torch.clone(r)
    for k in range(0,maxiter):
        Ap = jac_vec_prod(p)

        alpha = calc_alpha(done,r,p,Ap)
        x = x + alpha*p
        rplus = r - alpha*Ap
        
        done = calc_done(rplus,atol_sq)
        if torch.all(done):
            break
            

        beta = calc_beta(done,rplus,r)

        p = torch.clone(rplus+beta*p)
        r = torch.clone(rplus)
    return x

from torch_jacobian import jac_vec_prod,fn
def sqrt_mat_cg(X,resid,maxiter=10,atol=1e-16):
    return matcg(lambda xx: jac_vec_prod(X,xx),-resid,maxiter=maxiter,atol=atol)
    
if __name__=="__main__":
    
    from jacobian import get_jacobian
    from scipy.sparse.linalg import cg
    import numpy as np
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    N=20
    Y=torch.randn(N,N)
    Y=Y.T@Y
    
    X=torch.randn(N,N)
    X=X.T@X
    X=torch.cat([torch.eye(N).reshape(1,N,N),X.reshape(1,N,N)],0)
    Y=torch.cat([Y.reshape(1,N,N),Y.reshape(1,N,N)],0)
    
    
    
    
    Apfn = lambda x: jac_vec_prod(X,x)

    x1=matcg(Apfn,-fn(X,Y),maxiter=400,atol=1e-9).cpu().numpy()
    
    idx=1
    X=X[idx]
    Y=Y[idx]
    A=get_jacobian(X.cpu().numpy())
    b = -fn(X,Y).cpu().numpy().ravel()
    x2,info =cg(A,b,maxiter=400,tol=0.0,atol=1e-9)
    print(x2.reshape(N,N))
    print(x1[idx])
    print(np.linalg.norm(x2.reshape(N,N)-x1[idx]))