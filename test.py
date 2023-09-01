import unittest
from pyscf import gto
import src.ao_basis as ao_basis
from functools import reduce
import numpy as np

# toy settings
sys_dir   = "system/"
systems   = ["hydrogen.dat", "dihydrogen.dat", "water.dat"]
all_basis = ["sto-3g", "toy_basis", "aug-cc-pvdz", "aug-cc-pvtz"]
charge    = 0

# all integral symmetry options
perm_sym    = ["s1", "s2"]

class Test(unittest.TestCase):

    def test_ovlp_matrix(self):

        for atom in systems:

            system = sys_dir + atom
            print("testing %s" %system)
            spin = 0
            if atom == "hydrogen.dat":
                spin = 1

            for basis in all_basis:

                for aosym in perm_sym:
                        
                    # Create PySCF object
                    basname = ao_basis.format_name(basis)
                    mol = gto.M(atom = system, basis = basname, spin=spin,
                                charge=charge)

                    # Read basis to internal format
                    bas, Tcoeff = ao_basis.extract_ao(mol)

                    # Compute overlap integral
                    mat = ao_basis.int1e_ovlp(bas, aosym=aosym)

                    # transform matrix back to original PySCF basis using coeff
                    s = reduce(np.dot, (Tcoeff.T, mat, Tcoeff))

                    # Applying Tcoeff should get us back to PySCF overlap
                    is_same = np.allclose(mol.intor('int1e_ovlp'), s)
                    self.assertTrue(is_same)

if __name__ == '__main__':
    unittest.main()





