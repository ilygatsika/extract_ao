import numpy as np
from scipy.special import factorial2
from functools import lru_cache

"""
    One-electron two-center and four-center analytical overlap integrals for
    primitive Cartesian Gaussian type-orbitals
"""

##########################
#### Helper functions ####
##########################

def gauss_product(a,A,b,B):
    """
    Gaussian product theorem in 3D
    """
    p = a + b
    q = np.exp( - a*b/p * np.linalg.norm(A-B)**2)
    P = (a*A + b*B)/p
    return (q,p,P)

def choose(n,k):
    """
    A fast way to calculate binomial coefficients by Andrew Dalke (contrib).
    original code: https://stackoverflow.com/a/3025547
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0

def cache_pascal(n):
    """
    return binomial coefficients of binomial formula in 
    lower triangular matrix
    """
    N = n + 1
    mat = np.asarray([ [ choose(n,k) for n in range(N) ] for k in range(n+1) ], dtype=int)
    return mat

def HC(i,j,t,a,b,Q):    
    """
    Coefficients of the two-center Gaussian overlap distribution on a
    Hermite-Gaussian basis (Section 10.3, Chapter 12 Helgaker & Taylor)

    reference implementation:
    github.com/jjgoings/McMurchie-Davidson/blob/master/mmd/integrals/reference.py

    p,q,Q results of Gaussian theorem
    """
   
    p = a + b
    q = a*b/p
    if (t < 0) or (t > i+j):
        # out of bounds
        return 0.0
    elif i == j == t == 0:
        # base case
        return np.exp(-q*Q**2)
    elif i > 0:
        # decrease i
        e1 = HC(i-1,j,t-1,a,b,Q)
        e2 = HC(i-1,j,t  ,a,b,Q)
        e3 = HC(i-1,j,t+1,a,b,Q)
        return (0.5/p)*e1 - q*Q/a*e2 + (t+1)*e3
    elif j > 0:
        # decrease j
        e1 = HC(i,j-1,t-1,a,b,Q)
        e2 = HC(i,j-1,t  ,a,b,Q)
        e3 = HC(i,j-1,t+1,a,b,Q)   
        return (0.5/p)*e1 + q*Q/b*e2 + (t+1)*e3

@lru_cache(maxsize=20)
def _cache_HC(i,j,a,b,Q):
    return HC(i,j,0,a,b,Q)

##########################
## 1d overlap integrals ##
##########################

def gaussian(a,i):
    """ 
    1d Gaussian integrated over the whole space
    """

    if i % 2: 
        return 0
    else: 
        return np.sqrt(np.pi) * pow(2,-0.5*i) * \
                factorial2(i-1) * pow(a, -0.5*(i+1))

def gaussian_prod(cbinom, la, lb, PAx, PBx, p):
    """
    product of 1d Gaussians integrated over the whole space
    using binomial expansion
    """

    if ( la == lb == 0 ):
        return gaussian(p, 0)

    val = 0
    for ia in range(la + 1):
        ca = cbinom[ia,la]
        pa = PAx ** (la - ia)
        for jb in range(lb + 1):
            cb = cbinom[jb,lb]
            pb = PBx ** (lb - jb)

            l = ia + jb
            val += ca * cb * pa * pb * gaussian(p, l)

    return val

##########################
## 3d overlap integrals ##
##########################

def gaussian_3d(shell1,a,shell2,b):

    l1,m1,n1 = shell1
    l2,m2,n2 = shell2

    l, m, n = int(l1+l2), int(m1+m2), int(n1+n2)
    p = a + b
    vx = gaussian(p,l)
    vy = gaussian(p,m)
    vz = gaussian(p,n)

    # combine dimensions
    return vx * vy * vz

def gaussian_prod_3d(cbinom,shell1,a,A,shell2,b,B):

    l1,m1,n1 = shell1
    l2,m2,n2 = shell2

    # Gaussian product theorem
    q,p,P = gauss_product(a,A,b,B)
    PA, PB = P - A, P - B

    # apply binomial theorem for every dimension
    vx = gaussian_prod(cbinom, l1, l2, PA[0], PB[0], p)
    vy = gaussian_prod(cbinom, m1, m2, PA[1], PB[1], p)
    vz = gaussian_prod(cbinom, n1, n2, PA[2], PB[2], p)

    # combine dimensions
    return q * vx * vy * vz

def mmd(shell1,a,A,shell2,b,B):

    l1,m1,n1 = shell1
    l2,m2,n2 = shell2

    t = 0
    vx = HC(l1,l2,t,a,b,A[0]-B[0])
    vy = HC(m1,m2,t,a,b,A[1]-B[1])
    vz = HC(n1,n2,t,a,b,A[2]-B[2])

    return (np.pi/(a+b))**1.5 * vx * vy * vz

