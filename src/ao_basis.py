import src.wrapper as wrapper
import src.intor as intor
from multiprocessing import cpu_count
from pyscf import ao2mo
import pymp
import scipy
import numpy as np

""" 
    Routines for Cartesian Gaussian-type atomic orbital basis.
    AOs are stored in internal format.

    How to use:
    i-th primitive bf stored as
    -----------------------------------
    bas[i,0:3]  shell as integer tiplet
    bas[i,3]    orbital exponent
    bas[i,4]    coefficient
    bas[i,5]    index of atom
    bas[i,6:9]  center in Bohr
    -----------------------------------
"""

# for bas
PTR_SHELL = 0
PTR_EXP   = 3
PTR_COEFF = 4
ATOM_OF   = 5
PTR_COORD = 6
BAS_SLOTS = 9

# bohr to angstrom conversion
B2A = 0.529177249

# number of cores for parallel computation
NC = 2*cpu_count()

format_name = wrapper.format_name

def extract_ao(mol):
    """
    input:    mol     pyscf object
    output:   bas     internal Cartesian AO format
              Tcoeff  coefficients that recover mol 
    """

    if mol.cart:
        # simply decontract the basis
        mol_raw, coeff = mol.decontract_basis()

    else:
        # decontract and transform to Cartesian
        mol_raw, coeff = mol.to_uncontracted_cartesian_basis()

    bas = wrapper.extract_ao(mol_raw)
    Tcoeff = scipy.linalg.block_diag(*coeff)

    assert( bas.shape[0] == Tcoeff.shape[0] )

    return (bas, Tcoeff)

def atom_coords(bas, ib, unit="Bohr"):
    """
    unit is Angstrom or Bohr
    """

    if unit=="Bohr":
        return bas[ib,PTR_COORD:PTR_COORD+3]
    else: 
        return B2A*bas[ib,PTR_COORD:PTR_COORD+3]

def get_bf(bas, ib):
    """
    Get bf of index ib in internal format
    """

    l,m,n = bas[ib,PTR_SHELL:PTR_SHELL+3]
    expon = bas[ib,PTR_EXP]
    coeff = bas[ib,PTR_COEFF]
    ia    = bas[ib,ATOM_OF]
    
    return (l,m,n,expon,coeff,ia)

def as_string(bas, unit="Bohr"):
    
    nbas = bas.shape[0]
    string = "Total number of Cartesian primitives: %i\n" %nbas
    for ib in range(nbas):

        l,m,n,expon,coeff,ia = get_bf(bas, ib) 
        x,y,z = atom_coords(bas, ib, unit=unit)

        string += "AO %i:\t" %ib
        string += "shell (%i,%i,%i)\t" %(l,m,n)
        string += "expon %5.2e\t" %expon
        string += "coeff %5.2e\t" %coeff
        string += "at atom %i  " %ia
        string += "with coord (%.3f,%.3f,%.3f) %s\n" %(x,y,z,unit[0])

    return string

##########################
#### basis integrals  ####
##########################

def int1e_ovlp_per_pair(bas, pair):
    """
    bas     basis data in internal format
    pair    2-element list of bf indices

    return the value of the integral
    """
    
    ib, jb = pair

    # params of bf ib
    shelli = bas[ib,PTR_SHELL:PTR_SHELL+3]
    exponi = bas[ib,PTR_EXP]              
    coeffi = bas[ib,PTR_COEFF]            
    posi   = bas[ib,PTR_COORD:PTR_COORD+3]

    # params of bf jb
    shellj = bas[jb,PTR_SHELL:PTR_SHELL+3]
    exponj = bas[jb,PTR_EXP]              
    coeffj = bas[jb,PTR_COEFF]
    posj   = bas[jb,PTR_COORD:PTR_COORD+3] 
   
    n = int(max(max(shelli),max(shellj)))
    cbinom = intor.cache_pascal(n)

    val = intor.gaussian_prod_3d(
                    cbinom,            \
                    shelli,exponi,posi,\
                    shellj,exponj,posj \
                    )
    
    return coeffi * coeffj * val

def int1e_ovlp(bas, aosym='s1'):
    """
    bas      basis data in internal format of size nbas
    aosym    permutational symmetry (s1: none, s2: 2-fold ij=ji)

    return overlap matrix of dimension (nbas, nbas)
    """
    
    # cache pascal triangle
    nbas = bas.shape[0]
    maxL = np.max([max(bas[ib, PTR_SHELL:PTR_SHELL+3])**2 \
            for ib in range(nbas)])
    cbinom = intor.cache_pascal(int(maxL))

    # apply permutational symmetry to indices
    if aosym == 's1': 
        idx = np.asarray([(i,j) for i in range(nbas) for j in range(nbas)])
    elif aosym == 's2':
        idx = np.asarray([(i,j) for i in range(nbas) for j in range(i+1)])
    else: 
        raise ValueError("integral permutation symmetry not found")

    nidx = idx.shape[0]
    res = pymp.shared.array(nidx, dtype='float')
    
    with pymp.Parallel(NC) as p:
        for z in p.range(nidx):
            
            ib, jb = idx[z]

            # params of bf ib
            shelli = bas[ib,PTR_SHELL:PTR_SHELL+3].astype(int)
            exponi = bas[ib,PTR_EXP]              
            coeffi = bas[ib,PTR_COEFF]            

            # params of bf jb
            shellj = bas[jb,PTR_SHELL:PTR_SHELL+3].astype(int)
            exponj = bas[jb,PTR_EXP]              
            coeffj = bas[jb,PTR_COEFF]            

            val = 0
            
            if bas[ib, ATOM_OF] == bas[jb, ATOM_OF]:
                # orbitals have the same center
                val = intor.gaussian_3d(shelli,exponi,shellj,exponj)

            else: 
                # must bring orbitals to the same center
                posi = bas[ib,PTR_COORD:PTR_COORD+3]
                posj = bas[jb,PTR_COORD:PTR_COORD+3]

                #val = intor.mmd(shelli,exponi,posi,shellj,exponj,posj)
                val = intor.gaussian_prod_3d(
                        cbinom,            \
                        shelli,exponi,posi,\
                        shellj,exponj,posj \
                        )

            res[z] = coeffi * coeffj * val
                
    res = np.asarray(res)

    if aosym == 's1': 
        mat = res.reshape(nbas, nbas)
    elif aosym == 's2':
        # symmetrize
        mat = np.zeros((nbas, nbas), dtype=float)
        mat[np.tril_indices(nbas)] = res
        mat += mat.T - np.diag(np.diag(mat))

    return mat

def int4c1e(bas, aosym='s1'):     
    """
    bas      basis data in internal format of size nbas
    aosym    permutational symmetry (s1: none, s4: 4-fold ij=ji and kl=lk)

    return overlap matrix of dimension (nbas, nbas, nbas, nbas)
    """

    # cache pascal triangle
    nbas = bas.shape[0]
    maxL = np.max([max(bas[ib, PTR_SHELL:PTR_SHELL+3])**4 \
            for ib in range(nbas)])
    cbinom = intor.cache_pascal(int(maxL))

    # apply permutational symmetry to indices
    if aosym == 's1': 
        idx = np.asarray([(i,j,k,l)                       \
                for i in range(nbas) for j in range(nbas) \
                for k in range(nbas) for l in range(nbas) \
                ])
    elif aosym == 's4':
        idx = np.asarray([(i,j,k,l)                       \
                for i in range(nbas) for j in range(i+1)  \
                for k in range(nbas) for l in range(k+1)  \
                ])
    else: 
        raise ValueError("integral permutation symmetry not found")

    nidx = idx.shape[0]
    res = pymp.shared.array(nidx, dtype='float')
    
    with pymp.Parallel(NC) as p:
        for z in p.range(nidx):
            
            ib, jb, kb, lb = idx[z]

            # params of bf ib
            shelli = bas[ib,PTR_SHELL:PTR_SHELL+3].astype(int)
            exponi = bas[ib,PTR_EXP]              
            coeffi = bas[ib,PTR_COEFF]            

            # params of bf jb
            shellj = bas[jb,PTR_SHELL:PTR_SHELL+3].astype(int)
            exponj = bas[jb,PTR_EXP]              
            coeffj = bas[jb,PTR_COEFF]      

            # params of bf kb
            shellk = bas[kb,PTR_SHELL:PTR_SHELL+3].astype(int)
            exponk = bas[kb,PTR_EXP]              
            coeffk = bas[kb,PTR_COEFF]            

            # params of bf lb
            shelll = bas[lb,PTR_SHELL:PTR_SHELL+3].astype(int)
            exponl = bas[lb,PTR_EXP]              
            coeffl = bas[lb,PTR_COEFF]

            # prepare first ao pair
            posi = bas[ib,PTR_COORD:PTR_COORD+3]
            posj = bas[jb,PTR_COORD:PTR_COORD+3]
            q1,p1,P1 = intor.gauss_product(exponi,posi,exponj,posj)
            PA1, PA2 = P1 - posi, P1 - posj
            
            # prepare second ao pair
            posk = bas[kb,PTR_COORD:PTR_COORD+3]
            posl = bas[lb,PTR_COORD:PTR_COORD+3]
            q2,p2,P2 = intor.gauss_product(exponk,posk,exponl,posl)
            PB1, PB2 = P2 - posk, P2 - posl

            # apply 1d mmd
            vd = np.zeros(3,dtype=float)
            for d in range(3):
                for i1 in range(shelli[d] + 1):
                    c1 = cbinom[i1,shelli[d]] * PA1[d] ** (shelli[d] - i1)
                    for j1 in range(shellj[d] + 1):
                        c2 = cbinom[j1,shellj[d]] * PA2[d] ** (shellj[d] - j1)
                        for i2 in range(shellk[d] + 1):
                            c3 = cbinom[i2,shellk[d]] * PB1[d] ** (shellk[d] - \
                                    i2)
                            for j2 in range(shelll[d] + 1):
                                c4 = cbinom[j2,shelll[d]] * PB2[d] **          \
                                        (shelll[d] - j2)
                                
                                vd[d] += (c1*c2*c3*c4) *                       \
                                         (np.pi/(p1+p2))**0.5 *                \
                                         intor.HC(i1+j1,i2+j2,0,p1,p2,P1[d]-P2[d])
        
            #intor._cache_HC.cache_clear()

            # combine dimensions
            val = q1 * q2 * vd[0] * vd[1] * vd[2]

            res[z] = coeffi * coeffj * coeffk * coeffl * val
                
    res = np.asarray(res)

    if aosym == 's1': 
        mat = res.reshape(nbas, nbas, nbas, nbas)
    elif aosym == 's4':
        # symmetrize
        mat = ao2mo.restore(4, res, nbas)

    return mat






