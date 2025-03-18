from pyscf import gto, scf, ao2mo
import numpy as np


N_hydrogen = 2
nuclear_distance = 1

H_chain_string = "; ".join([f"H 0 0 {nuclear_distance*i}" for i in range(N_hydrogen)])
print(H_chain_string)

mol = gto.M(
    atom = H_chain_string,  # Adjust bond length if necessary
    basis = 'cc-pVDZ',
)

mf = scf.RHF(mol)
mf.kernel()  # Run HF calculation

from pyscf import fci
fci_h2 = fci.FCI(mf)
e_fci = fci_h2.kernel()[0]
print(e_fci)

mo_coeff = mf.mo_coeff 
N_orbitals = len(mf.mo_occ)
occ = mf.mo_occ# Molecular orbital coefficients

#Implement Cuthill–McKee algorithm for orbital ordering


from dmrg.utils import get_initial_states_from_mol_orb_occ
init_states = get_initial_states_from_mol_orb_occ(mf.mo_occ)
#print(init_states)

h1e = mf.get_hcore()  # One-electron integrals
h2e = ao2mo.kernel(mol, mo_coeff,aosym = "s1").reshape((N_orbitals,N_orbitals, N_orbitals, N_orbitals))  # Two-electron integrals
#print(h1e.shape)
#print(np.diag(h1e))
#print(h2e.shape)
