from torch_cg import sqrt_mat_cg
from torch_jacobian import fn
import torch
def mat_sqrt_newton(Y,Xinit,maxiter=100,miniter=2,tol=1e-4,tol_cg2=0.12,cg_iter=2,cg_atol=1e-16):
    N=Y.size()[-1]
    if Xinit is not None:
        assert(Xinit.size()==Y.size())
    else:
        Xinit = torch.eye(N)
    X=Xinit
    
    Ymean = torch.mean(Y,dim=(-2,-1),keepdim=True)
    Yscale = torch.mean((Y-Ymean)**2,dim=(-2,-1),keepdim=True)
    for i in range(0,miniter):
        dX = sqrt_mat_cg(X/Yscale,fn(X,Y)/Yscale,maxiter=1,atol=cg_atol)
        X=X+dX
        
    cgit=1
    for i in range(miniter,maxiter):
        resid = fn(X,Y)/Yscale
        
        err = torch.max(torch.mean(resid**2,dim=(-2,-1)))
        print(i,err ,cgit)
        if  err <= tol**2:
            break
        elif err <= tol_cg2**2: # If below this limit, increase number of cg_steps.
            cgit=cg_iter
            
        dX = sqrt_mat_cg(X/Yscale,resid,maxiter=cgit,atol=cg_atol)
        X=X+dX
        
    return X
    
    
if __name__ == "__main__":
    
    import time
    #torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    M=200
    N=50
    X=torch.randn(30,13,M,N)
    C=torch.matmul(torch.transpose(X,3,2),X)/M
    
    w,q = torch.linalg.eigh(C)
    w=w.reshape(30,13,N,1)
    Csqrt = torch.matmul(q,w**0.5*torch.transpose(q,3,2))
    
    
    dX = torch.randn(30,13,1,N)
    C2 = C+torch.matmul(torch.transpose(dX,3,2),dX)
    
    X=mat_sqrt_newton(C2,Csqrt)
    
    w,q = torch.linalg.eigh(C2)
    w=w.reshape(30,13,N,1)
    C2sqrt = torch.matmul(q,w**0.5*torch.transpose(q,3,2))
    
    C2mean = torch.mean(C2sqrt,dim=(-2,-1),keepdim=True)
    C2scale = torch.mean((C2sqrt-C2mean)**2,dim=(-2,-1),keepdim=True)
    print(torch.max(torch.sqrt(torch.mean((X-C2sqrt)**2,dim=(-2,-1),keepdim=True)/C2scale)))
    
    C2mean = torch.mean(C2,dim=(-2,-1),keepdim=True)
    C2scale = torch.mean((C2-C2mean)**2,dim=(-2,-1),keepdim=True)
    print(torch.max(torch.sqrt(torch.mean((torch.matmul(X,X)-C2)**2,dim=(-2,-1),keepdim=True)/C2scale)))
    
    