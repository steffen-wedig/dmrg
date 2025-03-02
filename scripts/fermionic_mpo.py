from dmrg.fermions.mpo import create_local_mpo_tensors
from pyscf import gto, scf, ao2mo
import numpy as np
from dmrg.initialization import single_site_operators

mol = gto.M(
    atom = 'H 0 0 0; H 0 0 1.1',  # Adjust bond length if necessary
    basis = 'cc-pVDZ',
    symmetry = True
)

mf = scf.RHF(mol)
mf.kernel()  # Run HF calculation
mo_coeff = mf.mo_coeff 
N_orbitals = len(mf.mo_occ)
h1e = mf.get_hcore()  # One-electron integrals


h2e_re = ao2mo.kernel(mol, mo_coeff,aosym = "s1").reshape((N_orbitals,N_orbitals, N_orbitals, N_orbitals)) 


h2e = ao2mo.kernel(mol, mo_coeff,aosym = "s1")

print(f"{h2e[0,0]} and {h2e_re[0,0,0,0]}")

print(np.prod(ao2mo.kernel(mol, mo_coeff).shape))

N_non_zero_h1e = np.sum(np.where(h1e == 0.0, 0, 1))
N_non_zero_h2e = np.sum(np.where(h2e == 0.0, 0, 1))

print(f"fraction of Non zero elements in the one electron integrals: {N_non_zero_h1e/np.prod(h1e.shape)}")
print(f"fraction of Non zero elements in the two electron integrals: {N_non_zero_h2e/np.prod(h2e.shape)}")



#print(h1e.shape)
one_e_mpo = create_local_mpo_tensors(h1e,h2e)
#remove the zero elementsö