# set python path
PYTHON=python3

SYSDIR=system/
MOLFILE=$(SYSDIR)$(MOL)

# example
MOL=water.dat
BAS=cc-pvdz
UNIT=Angstrom

help:
	@echo "usage: make <program> MOL=<string> UNIT=<string> BAS=<string>" 
	@echo "programs:"
	@echo "       extract_ao    print atomic orbital basis data in internal format" 
	@echo "       condition     compute condition number of basis from internal format" 
	@echo "       get_mo_coeff  transform MO coefficients from PySCF to the internal basis" 
	@echo "options:"
	@echo "       MOL           filename (including extension) of molecular geometry in XYZ format" 
	@echo "       UNIT          units of molecular geometry (Angstrom or Bohr)" 
	@echo "       BAS           name of atomic orbital basis (sto-3g, cc-pvdz, ..)" 
.PHONY: help

test: test.py
	$(PYTHON) $<
.PHONY: test

extract_ao: extract_ao.py
	$(PYTHON) $< $(MOLFILE) $(UNIT) $(BAS)
.PHONY: extract_ao

condition: condition.py
	$(PYTHON) $< $(MOLFILE) $(UNIT) $(BAS)
.PHONY: condition

get_mo_coeff: get_mo_coeff.py
	$(PYTHON) $< $(MOLFILE) $(UNIT) $(BAS)
.PHONY: get_mo_coeff

install:
	$(PYTHON) -m pip install -r requirements.txt
.PHONY: install


