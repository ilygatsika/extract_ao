extract_ao
==========

01-09-2023

A Python program for extracting atomic orbitals (AOs) of a Gaussian basis 
set from PySCF program package (https://github.com/pyscf/pyscf). Extracted AOs
are decomposed into Cartesian components and stored in internal format as polynomial 
times Gaussian functions in Cartesian coordinates.

The input of the program is a molecular geometry and any basis set in NWChem format. 
The output is a numpy array of :code:`nbas` rows and 9 columns containing parameters 
:code:`l,m,n,a,c,iatm,A` of GTO functions centered on the molecular geometry, 
of the following Cartesian form::

    c (x-A_x)^l (y-A_y)^m (z-A_z)^n exp(-a|r-A|^2)

The unit used in the internal format is Bohr (for compatibility with PySCF internal 
format). 

Requirements
------------

numpy, scipy, pyscf, pymp-pypi

How to use
----------

* Install requirements::

    make install

* Test suite::

    make test

* Get help::

    make help

* Run main functionality::

    make extract_ao

* Run applications::

    make condition
    make get_mo_coeff

Features
--------

* Provides coefficients for transforming Cartesian primitive functions back to
  the original PySCF basis functions (possibly spherical and/or contracted). This 
  can be useful for expressing the molecular orbital (MO) coefficients computed
  on the PySCF basis to the internal basis, as in get_mo_coeff.py.

* Uses the internal basis format for computing custom analytical molecular integrals.
  As a simple example, the two-center one-electron overlap integral
  is implemented for the basis set in internal format. The overlap matrix may be
  used to compute the condition number of the basis in condition.py. A (slow) 
  implementation of the four-center overlap integral is used in ovlp_4c.py.

* Add new molecules in "system" directory. 
  Required data format: XYZ

* Add new user-defined atomic orbital basis sets in "basis" directory. 
  Required data format: NWChem
  Required file extension: .dat

Contact
-------
Ioanna-Maria Lygatsika <ioanna-maria.lygatsika@sorbonne-universite.fr>


