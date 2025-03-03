from dmrg.fermions.mpo import create_local_mpo_tensors, reformat_mpo
from pyscf import gto, scf, ao2mo

from dmrg.einsum_optimal_paths import EinsumEvaluator


mol = gto.M(
    atom = 'H 0 0 0; H 0 0 1.1',  # Adjust bond length if necessary
    basis = '3-21G',
    symmetry = True
)

mf = scf.RHF(mol)
mf.kernel()  # Run HF calculation
mo_coeff = mf.mo_coeff 
N_orbitals = len(mf.mo_occ)
h1e = mf.get_hcore()  # One-electron integrals


h2e = ao2mo.kernel(mol, mo_coeff,aosym = "s1").reshape((N_orbitals,N_orbitals, N_orbitals, N_orbitals)) 

mpo = create_local_mpo_tensors(h1e,h2e,N_sites=N_orbitals)
mpo = reformat_mpo(mpo)
import numpy as np


einsum_eval = EinsumEvaluator()


res = einsum_eval("ijkl,jmno->imknlo",mpo[0],mpo[1])

res = einsum_eval("ijkl,jmno->imknlo",mpo[0],mpo[1])