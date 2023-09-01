from pyscf import gto
import numpy as np
from os.path import realpath, join

"""
    Wrapper routines for manipulating PySCF objects using PySCF functions
"""

# for bas
PTR_SHELL = 0
PTR_EXP   = 3
PTR_COEFF = 4
ATOM_OF   = 5
PTR_COORD = 6
BAS_SLOTS = 9

# cartesian labels
MAPM2SHELL = ['x','y','z']

def to_triplet(xyz):
    """
    converts label 'xxy' to 210, 
                   'yyz' to 021, ..
    """
    
    shell = [0,0,0]
    if not len(xyz.strip()): 
        return shell
    for s in xyz:
        shell[MAPM2SHELL.index(s)] += 1
    return shell

def format_name(basis):

    if gto.basis._format_basis_name(basis) in gto.basis.ALIAS:
        return basis
    else:
        dirnow = realpath(join(__file__,'../../basis/'))
        user_basis = join(dirnow, basis+'.dat')
        return user_basis
        
def extract_ao(mol):
    """
    input is mol (spherical or Cartesian)
    Read primitive basis from PySCF in Cartesian format
    """

    # test input
    for ib in range(mol.nbas):
        if not len(mol._libcint_ctr_coeff(ib)) == 1:
            raise ValueError("basis should be uncontracted")

    # extract data
    coords = mol.atom_coords(unit='Bohr') # use internal PySCF units
    bas = np.empty((mol.nao,BAS_SLOTS), dtype=float)
    jc = 0
    for ib in range(mol.nbas):
        
        # coefficient (used in integrals of PySCF)
        coeff = mol._libcint_ctr_coeff(ib)[0]
        
        # orbital exponent
        expo = mol.bas_exp(ib)[0]

        # atomic position
        ia = mol.bas_atom(ib)
        pos = coords[ia]

        # number of cartesian components
        l = mol.bas_angular(ib)
        ncart = (l+1) * (l+2) // 2 

        # empirical correction for PySCF internal coeff 
        # (see github.com/sunqm/libcint/issues/22)
        if l < 2: coeff = coeff * gto.mole.cart2sph(l)[0][0]

        # shell in triplet format
        # 0: s, 1: p, 2: d, ..
        for ic in range(ncart):

            assert(mol.cart_labels(0)[jc][0] == ia)
            xyz = mol.cart_labels(0)[jc][-1]
            shell = to_triplet(xyz)
            assert(sum(shell) == l)

            # store
            bas[jc,PTR_SHELL:PTR_SHELL+3] = shell
            bas[jc,PTR_EXP]               = expo
            bas[jc,PTR_COEFF]             = coeff
            bas[jc,ATOM_OF]               = ia
            bas[jc,PTR_COORD:PTR_COORD+3] = pos
            jc += 1
        
    return bas


    


