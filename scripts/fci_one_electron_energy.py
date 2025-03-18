import numpy as np
from pyscf import gto, scf

# Define a molecule (example: H2 molecule)
mol = gto.M(
    atom = '''
      H 0 0 0
      H 0 0 0.74
    ''',
    basis = 'sto-3g'
)
mol.build()

# Run an RHF calculation
mf = scf.RHF(mol)
mf.kernel()

# Get the one-electron (core) Hamiltonian in the atomic orbital (AO) basis
hcore = mf.get_hcore()

# For a closed-shell system, the density matrix in the AO basis is given by:
#   D = 2 * C_occ * C_occ^T,
# where C_occ are the occupied molecular orbital coefficients.
nocc = mol.nelectron // 2
C_occ = mf.mo_coeff[:, :nocc]
D = 2 * np.dot(C_occ, C_occ.T)

# Compute the one-electron CI energy using the density matrix:
E_CI = np.einsum('ij,ji', hcore, D)

print(E_CI)

mo_energy = mf.mo_energy
E_CI_alt = 2 * np.sum(mo_energy[:nocc])
print("One-electron CI energy (using MO eigenvalues):", E_CI_alt)