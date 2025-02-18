from pyscf import gto, scf, ao2mo
import numpy as np


mol = gto.M(
    atom = 'N 0 0 0; N 0 0 1.1',  # Adjust bond length if necessary
    basis = 'cc-pVDZ',
    symmetry = True
)

mf = scf.RHF(mol)
mf.kernel()  # Run HF calculation

mo_coeff = mf.mo_coeff  # Molecular orbital coefficients
print(mo_coeff.shape)

h1e = mf.get_hcore()  # One-electron integrals
h2e = ao2mo.kernel(mol, mo_coeff)  # Two-electron integrals
print(h1e.shape)
print(h2e.shape)
