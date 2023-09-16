import numpy as np
from jacobian import get_jacobian,cg_step
from scipy.sparse.linalg import cg
import time
def fn(X,Y):
    return X@X-Y
    
def_cg_opts = {"maxiter":None,"tol":1e-6,"atol":None}
def newton(Y,Xinit=None,max_iter=100,tol=1e-6,cg_opts=def_cg_opts):
    N=Y.shape[0]
    if Xinit is not None:
        assert(np.all(Xinit.shape==Y.shape))
    else:
        Xinit = np.eye(N)
    X=Xinit
    for i in range(0,max_iter):

        
                    
        A,b = get_jacobian(X), -fn(X,Y).ravel()
        dX, info = cg(A,b,maxiter=cg_opts["maxiter"],tol=cg_opts["tol"],atol=cg_opts["atol"])
        dX = dX.reshape(N,N)            

        X=X+dX
        if np.linalg.norm(fn(X,Y))<=tol:
            break
        
    return X
    

def sqiter(Y,Xinit=None,max_iter=100,tol=1e-6):
    N=Y.shape[0]
    if Xinit is not None:
        assert(np.all(Xinit.shape==Y.shape))
    else:
        Xinit = np.eye(N)
    X=Xinit
    for i in range(0,max_iter):

        
        dX=cg_step(X,-fn(X,Y))
        X=X+dX
        if tol is not None:
            if i%100==0:
                print(i,np.linalg.norm(fn(X,Y)))
            if np.linalg.norm(fn(X,Y))<=tol:
                break
        
    return X
    

if __name__=="__main__":
    def test1():
        import time
        np.random.seed(0)
        M=1000
        N=100
        
        C=0.0
        for i in range(0,10):
            X=np.random.randn(M,N)
            C+=X.T@X
            print(i)
        
        dX = np.random.randn(2,N)
        dC = dX.T@dX
        
        C2 = C+dC
        
        
        w,q = np.linalg.eigh(C)
        xinit = q@np.diag(w**0.5)@q.T    
        
        t0 = time.time()
        w,q = np.linalg.eigh(C2)
        sq2 = q@np.diag(w**0.5)@q.T
        t2 = time.time()-t0
        print("EIGH: ",np.linalg.norm(sq2-sq2), t2)
        
        w,q = np.linalg.eigh(sq2-xinit)
        
        print("INIT: ",np.linalg.norm(xinit-sq2), 0)
        
        #xinit=1e9*np.eye(N)#.ravel()
        t0 = time.time()
        sq1 = sqiter(C2,Xinit=xinit,max_iter=2)
        t1 = time.time()-t0
        print("NEWTON: ",np.linalg.norm(sq1-sq2), t1)
    
    def test2():
        A = np.array([[10,7,8,7],[7,5,6,5],[8,6,10,9],[7,5,9,10]])
        
        w,q = np.linalg.eigh(A)
        sq1=q@np.diag(w**0.5)@q.T
        
        sq2 = sqiter(A,max_iter=100)

        def bab(A,niter):
            Y=np.eye(A.shape[0])
            for i in range(0,niter):
                Y = 0.5*(Y+np.linalg.inv(Y)@A)
            return Y
        sq3 = bab(A,10)
        print(sq1)
        print(sq2)
        print(sq3)
        
    def test3():
        N=500
        X=np.random.randn(N,2)
        K=np.zeros((N,N))
        sig=10.0
        for i in range(0,N):
            K[i,:] = np.exp(-0.5*np.sum(((X[i]-X)/sig)**2,axis=1))
        K+=np.eye(N)*1e-10
        w,q = np.linalg.eigh(K)
        sq1=q@np.diag(w**0.5)@q.T
        sq2=newton(K,max_iter=100,cg_opts = {"maxiter":100,"tol":1e-10,"atol":None})

        print(sq1)
        print(sq2)
    test1()
   