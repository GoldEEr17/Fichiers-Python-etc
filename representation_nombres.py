""" TP1 Representation binaire des nombres : outils"""

import numpy as np
import struct


def _bin_repr_entier(number):
    """
    Renvoie le str de la representation binaire du uint8 ou int8 donné
    """
    return np.binary_repr(number, width=8)


def _bin_repr_float32(number):
    """
    Renvoie le str de la representation binaire du float32 donné
    """
    float_bytes = struct.pack("f", number)
    byte_list = [format(b, "08b") for b in float_bytes]
    byte_list.reverse()
    return " ".join(byte_list)


def repr_binaire(nombre):
    """
    Recoit un uint8, un int8, ou un float32, et renvoie la chaine de caractères de sa representation binaire.
    """
    if type(nombre) in (np.int8, np.uint8):
        return _bin_repr_entier(nombre)
    elif type(nombre) == np.float32:
        return _bin_repr_float32(nombre)
    else:
        raise TypeError("Type de nombre {} non pris en charge par la fonction de representation ci-presente. Seulement uint8, int8, et float32 (numpy)")




# print (0.1+0.1+0.1==0.3)round ( 0 . 1 , r ) + round ( 0 . 1 , r ) + round ( 0 . 1 , r ) = = round ( 0 . 3 , r )
# r=11

# print( round ( 0.1 , r ) + round ( 0.1 , r ) + round ( 0.1 , r ) == round ( 0.3 , r ) )

# print(round ( 0.1 + 0.1 + 0.1 , r ) == round ( 0.3 , r ) )
#
# r= 3
# a = round(1,001, r)

def fac(n) :
    return 1 if n==0 else n*fac(n-1)
# def f1():
#     s = 0
#     for n in range(0,30+1):
#         s += (-10)**n / fac(n)
#
#     return s
#
#
# def f2() :
#     s = 0
#     for n in range(0,30+1):
#         s += 10**n / fac(n)
#
#     return 1/s
#
#
# def f3():
#     s = 0
#     k = 9
#     for n in range(1,10**k):
#         s += 1/n
#     return s
#
# def f4():
#     s = 0
#     k = 9
#     for n in range(1,10**k):
#         s += 1/(10**k-n)
#     return s
#
# from math import log
#
# def integrale_i(k):
#     if k==0:
#         return(log(11/10))
#     else:
#         return((1/k)-10*integrale_i(k-1))
#
# def estimation(k):
#     return 1/(10*(k+1))
#
#
# def retour_rec(k,x):
#     assert k>=1
#     return((1/10)*(1/k-x))
#
# def retour():
#     k=50
#     x=estimation(k)
#     for i in range(30):
#         x=retour_rec(k,x)
#         k=k-1
#     return(x)


#
# for k in range(1,30):
#     print(k, integrale_i(k))



from scipy.integrate import quad

def int_calc(k):
    def f_k(x):
        return x**k / (10+x)

    return quad(f_k, 0, 1)










def der(f, x, h):
    return ( f(x+h)-f(x) ) / h


def f1(x):
    return np.sin(3*x)



import matplotlib.pyplot as plt

def graphe_erreurs():

    Krange = list(range(1,15+1))
    H = [10**-k for k in Krange]
    D = [ der(f1, 1, h) for h in H]
    logH = [-log(h,10) for h in H]
    # logD = [-log(-h,10) for h in D]
2 |f ′′(η)|

    plt.plot(logH, D)
    plt.show()

    # for k in range(15+1):
    #     print( 10**-k,  der(f1, 1, 0.01))



import sys
eps = sys.float_info.epsilon
















































































