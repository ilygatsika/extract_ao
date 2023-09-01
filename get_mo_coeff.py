from pyscf import gto, scf
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

# Run Koh-Sham calculation with PySCF
mf = scf.RKS(mol)
mf.xc = 'b3lyp'
E_gs = mf.kernel()

# Get density matrix
dm = mf.make_rdm1()

# Get mo_coeff
mo_coeff = mf.mo_coeff

# Transform density matrix and mo_coeff to internal basis
T_dm = Tcoeff @ dm @ Tcoeff.T
T_mo_coeff = Tcoeff @ mo_coeff

# Verify the internal density matrix is correct by computing electron number 
mat = ao_basis.int1e_ovlp(bas, aosym='s2')
nelec = np.trace(mat @ T_dm)

# verify it is the same as PySCF
print("PySCF number of electrons %f" %np.trace(mol.intor('int1e_ovlp') @ dm))
print("Our number of electrons   %f" %nelec)





