from scipy.optimize import newton_krylov
import numpy as np

def solve_diff_eq(l, n, f, mu1, mu2, max_iter=100, tol=1e-6):
    h = l / (n + 1)
    x = np.linspace(h, l - h, n)
    u = np.zeros(n + 2)
    u[0] = mu1
    u[-1] = mu2

    def F(u):
        F = np.zeros(n)
        for i in range(n):
            F[i] = (-u[i+2] + 2*u[i+1] - u[i]) / h**2 - f(x[i], u[i+1])
        return F

    def J(u):
        J = np.zeros((n, n))
        for i in range(n):
            J[i, i] = -2 / h**2 - df_du(x[i], u[i+1])
            if i > 0:
                J[i, i-1] = 1 / h**2
            if i < n-1:
                J[i, i+1] = 1 / h**2
        return J

    u_newton = newton_krylov(F, u[1:-1], maxiter=max_iter, tol=tol, verbose=1, jac=J)
    u[1:-1] = u_newton
    return x, u[1:-1]

def f(x, u):
    return -np.exp(-u)

def df_du(x, u):
    return np.exp(-u)

x, u = solve_diff_eq(1, 100, f, 0, 0)
