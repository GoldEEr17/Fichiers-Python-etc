import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu
from scipy.linalg import cholesky
from scipy.linalg import cholesky_banded


L=5
def A(n):
    A = np.zeros((n,n))
    A += np.diag([2 for _ in range(n)],0)
    A += np.diag([-1 for _ in range(n-1)],-1)
    A += np.diag([-1 for _ in range(n-1)],+1)

    h = L/(n+1)
    A = 1/h**2 * A
    return A

def Lapl_base(n):
    A = np.zeros((n,n))
    A += np.diag([2 for _ in range(n)],0)
    A += np.diag([-1 for _ in range(n-1)],-1)
    A += np.diag([-1 for _ in range(n-1)],+1)
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

#%%

def lu_b_exo2():
    B = np.array([[-2,3,0],[1,0,-4],[2,0,5]])
    return lu(B)

def exo2_veriflu():
    L = np.array([[1,0,0],[-0.5,1,0],[-1,2,1]])
    U = np.array([[-2,3,0],[0,1.5,-4],[0,0,13]])
    return np.dot(L, U)

def exo2_verif_sys():
    B = np.array([[-2,3,0],[1,0,-4],[2,0,5]])
    x = 1/13*np.array([[22,19,-1]]).T
    print(x)
    return np.dot(B, x)

## TP n°3 - Séance 5 ###


def decomp_lapl_lu(n):
    return lu(Lapl_base(n))

def essai():
    return(np.array([[1,2,1],[2,13,-1],[1,-1,3]]))

def identite():
    return np.array([[1,0,0],[0,1,0],[0,0,1]])

## TP n)3 - Séance 6 ###

'''Exercice 5 '''

B = np.eye(3)
B[1,2] = 5
D = np.random.uniform(-5, 5, (3, 3))

b = np.array([[3],[2],[0]])
c = np.array([[3],[0],[0]])
d = np.array([[10],[1]])




def householder(v):
    (l,c)=np.shape(v)
    return(np.identity(l)-2*(v*(v.T))/np.trace((v.T)*v))






#%%


def N(A):
    return np.trace(np.dot(A.T,A)) ** 0.5

def v(a):
    (n,_) = np.shape(a)
    s_2n = 0
    for i in range(1,n):
        s_2n += abs(a[i,0])

    if s_2n == 0 :
        return a

    elif s_2n > 0 :
        e1 = np.zeros_like(a)
        e1[0,0] = 1

        return a - N(a)*e1



def H(v):
    n,_ = np.shape(v)
    return np.eye(n) - 2*np.dot(v,v.T)/(N(v)**2)


def Hk(k,n,v):
    k = k-1
    HH = np.zeros((n,n))

    Ik = np.eye(k)
    Hv = H(v)

    HH[0:k,0:k] += Ik
    HH[k:n,k:n] += Hv

    return HH


def ak(k, n, HA): # k = le NOMBRE pas l'indice
    k = k-1
    return HA[k:n,k:k+1]


def householder_qr(A):
    n,n = np.shape(A)

    H = np.eye(n)
    HA = A.copy()

    for k in range(1,n):
        H_k = Hk(k,n,v(ak(k, n, HA)))

        HA = np.dot(H_k, HA)
        H = np.dot(H_k, H)

        # print(k)
        # print(H)
        # print(HA)

    Q, R = H.T, HA
    #
    # print( np.matmul(np.linalg.inv(H), HA) )
    # print( np.matmul(H.T, HA) )


    return Q,R


def verif():

    print(D)
    print()

    Q, R = householder_qr(D)
    print(Q)
    print(R)
    print()

    print(np.dot(Q,R))


def col(A,i):
    Vi = A[:,i].reshape(-1,1)
    return Vi

def qr_schmidt(A):
    assert np.linalg.det(A) != 0
    n,_ = np.shape(A)

    O = np.zeros((n,n))
    S = np.zeros((n,n))

    W = []

    for i in range(n):
        Vi = col(A,i)
        Wi = Vi.copy()
        for k in range(i):
            S[k,i] = np.dot(Vi,W[k])
            Wi -= S[k,i] * W[k]

        S[i,i] = nWi = np.linalg.norm(Wi)
        Wi = 1/nWi * Wi

        O[:,i] = W[i] = Wi.resize(-1,1)













