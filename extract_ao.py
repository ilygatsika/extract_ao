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

# Print a single basis function in internal format
print("Example of internal basis format \
(l,m,n, expon, coeff, index of atom):")
print(ao_basis.get_bf(bas,0))

# Print basis data
print(ao_basis.as_string(bas, unit=unit))


