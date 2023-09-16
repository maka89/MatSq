# MatSq
Iterative method to find square root of a matrix
- More numerically stable than babylonian method.
- Can make use of initial guess, as opposed to Denman beavers.
- Only matrix multiplications and suc. No inverses in the iteration.

  Uses Newton's method. Step is calculated using Conjugate Gradient, truncated to one or a few iterations.

  
## Theory

Want to solve [N,N] matrix-equation:

$$ F(X) = XX-Y = 0$$

Newtons method:

$$ F'(X) (X_{k+1}-X_k) = - f(X_k)$$

$F'(X)$ is a sparse [N,N,N,N] tensor. But we have a nice expression for the jacobian-vector product (Or jacobian-matrix product in this case?).

$$ F'(X)H = XH+HX $$

See [N.Higham - Newton's Method for the Matrix Square Root](https://www.ams.org/journals/mcom/1986-46-174/S0025-5718-1986-0829624-5/S0025-5718-1986-0829624-5.pdf)
