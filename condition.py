from pyscf import gto
import src.ao_basis as ao_basis
from functools import reduce
import numpy as np
import sys

# User input
system  = str(sys.argv[1])
unit    = str(sys.argv[2])
basis   = str(sys.argv[3])

# Create a PySCF mol
basis = ao_basis.format_name(basis)
mol = gto.M(atom = system, basis = basis, spin=0, charge=0, unit=unit)
# put spin=1 if system is Hydrogen

# Extract primitive Cartesian AO basis
bas, Tcoeff = ao_basis.extract_ao(mol)

# Compute overlap integral on internal basis
mat = ao_basis.int1e_ovlp(bas, aosym='s2')

# Transform overlap integral to PySCF basis
mat_s = reduce(np.dot, (Tcoeff.T, mat, Tcoeff))

print("Condition of Cartesian components of the basis: %3.2e" %np.linalg.cond(mat))

cc = "spherical"
if mol.cart==True: cc="Cartesian"
print("Condition of %s contracted basis: %3.2e" %(cc, np.linalg.cond(mat_s)))


