from pyscf import gto
import src.ao_basis as ao_basis
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

# Compute overlap integral
mat = ao_basis.int4c1e(bas, aosym='s4')



