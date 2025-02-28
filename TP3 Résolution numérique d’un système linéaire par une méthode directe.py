import numpy as np
import scipy.linalg


def lu_b_exo2():
    B = np.array([[-2,3,0],[1,0,-4],[2,0,5]])
    return scipy.linalg.lu(B)

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


# def decomp_lapl_lu(n):
#     return scipy.linalg.lu(Lapl_base(n))

def essai():
    return(np.array([[1,2,1],[2,13,-1],[1,-1,3]]))

def identite():
    return np.array([[1,0,0],[0,1,0],[0,0,1]])

## TP n°3 - Séance 6 ###

#%% EXERCICE 

# B = np.eye(3)
# B[1,2] = 5
b = np.array([[3],[2],[0]])
D = np.random.uniform(-5, 5, (3, 3))

# F = np.random.uniform(-5, 5, (5, 3))
# G = np.random.uniform(-5, 5, (1, 2))
# Z = np.random.uniform(-5, 5, (1, 3))
# K = np.random.uniform(-5, 5, (5, 2))
# F = np.random.uniform(-5, 5, (2, 2))
# G = np.random.uniform(-5, 5, (2, 2))
# Z = np.random.uniform(-5, 5, (2, 2))
# K = np.random.uniform(-5, 5, (2, 2))

# U = np.random.uniform(-5, 5, (3, 3))
# V = np.random.uniform(-5, 5, (3, 2))
# W = np.random.uniform(-5, 5, (2, 2))


# c = np.array([[3],[0],[0]])
# d = np.array([[10],[1]])




# def householder(v):
#     (l,c)=np.shape(v)
#     return(np.identity(l)-2*(v*(v.T))/np.trace((v.T)*v))






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
    return np.eye(n) - 2*np.dot(v,v.T) / N(v)**2


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


def verif_hqr():

    print(D)
    print()

    Q, R = householder_qr(D)
    print(Q)
    print(R)
    print()

    print(np.dot(Q,R))


def col(A,i, plat=False):
    if not plat :
        Vi = A[:,i].copy().reshape(-1,1)
        return Vi
    else :
        return A[:,i].copy()

# def qr_schmidt(A):
#     assert np.linalg.det(A) != 0
#     n,_ = np.shape(A)

#     O = np.zeros((n,n))
#     S = np.zeros((n,n))

#     W = [-1] * n

#     for i in range(n):
#         Vi = col(A,i)
#         Wi = Vi.copy()
#         for k in range(i):
#             S[k,i] = np.vdot(Vi,W[k])
#             Wi -= S[k,i] * W[k]

#         S[i,i] = nWi = np.linalg.norm(Wi) # double assignation jamais dangereuse car c'est une valeur non mutable
#         Wi = 1/nWi * Wi

#         # O[:,i] = W[i] = Wi.reshape(-1,1) # techniquement cette double assignation est dangereuse car O[:,i] et W[i] partagent alors la même référence mémoire vers un objet mutables donc seraient modifiés tous les deux. Mais dans ce code ce n'est pas prévu de les modifier et ça serait cohérent anyway.
#         O[:,i] = Wi.ravel()
#         W[i] = Wi.reshape(-1,1)
    
#     return O, S
        
def qr_schmidt(A):
    assert np.linalg.det(A) != 0
    n,_ = np.shape(A)

    O = np.zeros((n,n))
    S = np.zeros((n,n))

    W = [-1] * n

    for i in range(n):
        Vi = col(A,i, plat=True)
        Wi = Vi.copy()
        for k in range(i):
            S[k,i] = np.vdot(Vi,W[k])
            Wi -= S[k,i] * W[k]

        S[i,i] = nWi = np.linalg.norm(Wi) # double assignation jamais dangereuse car c'est une valeur non mutable
        Wi = 1/nWi * Wi

        O[:,i] = W[i] = Wi # techniquement cette double assignation est dangereuse car O[:,i] et W[i] partagent alors la même référence mémoire vers un objet mutables donc seraient modifiés tous les deux. Mais dans ce code ce n'est pas prévu de les modifier et ça serait cohérent anyway.
      
    
    return O, S

        
#%%

def verif_sqr():
    print(D)
    print()
    
    Q, R = qr_schmidt(D)
    print(Q)
    print(R)
    
    print()
    print(Q@R)

# qr_schmidt(D)
        
