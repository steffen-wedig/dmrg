from pyscf import gto, scf, ao2mo
import numpy as np


mol = gto.M(
    atom = 'N 0 0 0; N 0 0 1.1',  # Adjust bond length if necessary
    basis = 'cc-pVDZ',
    symmetry = True
)

mf = scf.RHF(mol)
mf.kernel()  # Run HF calculation

mo_coeff = mf.mo_coeff 
N_orbitals = len(mf.mo_occ)
occ = mf.mo_occ# Molecular orbital coefficients

#Implement Cuthill–McKee algorithm for orbital ordering


from dmrg.initialization import get_initial_states_from_mol_orb_occ
init_states = get_initial_states_from_mol_orb_occ(mf.mo_occ)
print(init_states)

h1e = mf.get_hcore()  # One-electron integrals
h2e = ao2mo.kernel(mol, mo_coeff,aosym = "s1").reshape((N_orbitals,N_orbitals, N_orbitals, N_orbitals))  # Two-electron integrals
print(h1e.shape)
print(np.diag(h1e))
print(h2e.shape)
