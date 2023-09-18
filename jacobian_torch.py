import torch
    
def fn(X,Y):
    return torch.matmul(X,X)-Y
    
def trans_jac_vec_prod(X,v):
    Xt=torch.transpose(X,-1,-2)
    return torch.matmul(Xt,v) + torch.matmul(v,Xt)
    
def jac_vec_prod(X,v):
    return torch.matmul(X,v)+torch.matmul(v,X)


def cg_step(X,B):

    alpha = torch.sum(B**2,dim=(-2,-1),keepdim=True)/torch.sum(B*jac_vec_prod(X,B),dim=(-2,-1),keepdim=True)
    return alpha*B

if __name__=="__main__":
    #torch.set_default_dtype(torch.float64)
    N=9
    steps=10
    X=torch.randn(256,256,N,N)
    Y=torch.randn(256,256,N,N)
    X=torch.matmul(torch.transpose(X,3,2),X)
    Y=torch.matmul(torch.transpose(Y,3,2),Y)
    X=X.to(torch.device("cuda"))
    Y=Y.to(torch.device("cuda"))
    import time
    t0 = time.time()
    X2 = torch.clone(X)
    for i in range(0,steps):
        X2 = cg_step(X2,Y)
    print(X2)
    print(time.time()-t0)
    answ_torch=X2[3,4].cpu().detach().numpy()
    
    
    from jacobian import cg_step
    import numpy as np
    X2 = np.copy(X[3,4].cpu().detach().numpy())
    Y=Y[3,4].cpu().detach().numpy()
    for i in range(0,steps):
        X2 = cg_step(X2,Y)
    answ_numpy=X2
    print(np.linalg.norm(answ_torch-answ_numpy))
        
    