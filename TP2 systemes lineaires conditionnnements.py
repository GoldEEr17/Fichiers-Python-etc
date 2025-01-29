import numpy as np
import matplotlib.pyplot as plt

L=5
def A(n):
    A = np.zeros((n,n))
    A += np.diag([2 for _ in range(n)],0)
    A += np.diag([-1 for _ in range(n-1)],-1)
    A += np.diag([-1 for _ in range(n-1)],+1)

    h = L/(n+1)
    A = 1/h**2 * A
    return A



def f1(x):
    return 1

def f2(x):
    sigma = 3
    return np.exp(-sigma*(x-L/2)**2)


def trace_sol(f,n):
    h = L/(n+1)
    X = [0+i*h for i in range(1,n+1)]
    Y = [f(xi) for xi in X]

    u = np.linalg.solve(A(n), Y)

    plt.plot(X,u)

    plt.show()


def vp_th(n):
    h = L/(n+1)

    valeur_p = np.array([2/h**2 * (1-np.cos(k*np.pi/(n+1))) for k in range(1,n+1) ])
    u = np.array([[np.sin(k*i*np.pi/(n+1)) for i in range(1,n+1)] for k in range(1,n+1)]).T

    return (valeur_p, u)

def vp_calc(n):
    return np.linalg.eig(A(n))



def trace_2_vp(n):
    h = L/(n+1)

    vp_op = [(k*np.pi/L)**2 for k in range(1,n+1)]
    vp_An = [2/h**2 * (1-np.cos(k*np.pi/(n+1))) for k in range(1,n+1) ]

    plt.plot(range(1,n+1), vp_op)
    plt.plot(range(1,n+1), vp_An)

    plt.show()

























