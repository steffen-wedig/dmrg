from dmrg.fermions.mpo import create_local_mpo_tensors, reformat_mpo
from pyscf import gto, scf, ao2mo
import numpy as np
from dmrg.initialization import single_site_operators
from dmrg.fermions.mps import get_mps_from_occupation_numbers, get_random_mps
from dmrg.einsum_optimal_paths import EinsumEvaluator

from dmrg.heisenberg_chain.mps import create_neel_mps, right_canonicalize
from dmrg.heisenberg_chain.mpo import initialize_heisenberg_mpo
from dmrg.heisenberg_chain.sweep import precompute_right_environment, right_to_left_sweep, left_to_right_sweep




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


h2e = ao2mo.kernel(mol, mo_coeff,aosym = "s1").reshape((N_orbitals,N_orbitals, N_orbitals, N_orbitals)) 
print(np.prod(ao2mo.kernel(mol, mo_coeff).shape))

N_non_zero_h1e = np.sum(np.where(h1e == 0.0, 0, 1))
N_non_zero_h2e = np.sum(np.where(h2e == 0.0, 0, 1))

print(f"fraction of Non zero elements in the one electron integrals: {N_non_zero_h1e/np.prod(h1e.shape)}")
print(f"fraction of Non zero elements in the two electron integrals: {N_non_zero_h2e/np.prod(h2e.shape)}")



#print(h1e.shape)
mpo = create_local_mpo_tensors(h1e,h2e,N_sites=N_orbitals)

#Reformat the MPO
mpo = reformat_mpo(mpo)

#remove the zero elementsö

from dmrg.initialization import get_initial_states_from_mol_orb_occ
init_states = get_initial_states_from_mol_orb_occ(mf.mo_occ)
print(init_states)

mps = get_mps_from_occupation_numbers(init_states,5)
mps = get_random_mps(L= len(mpo),bond_dimensions=5)

L = len(mps)

R_env = [None] * (L+1)

einsum_eval = EinsumEvaluator()

mps = right_canonicalize(mps, einsum_eval)

R_env = precompute_right_environment(mps,mpo,einsum_eval)
       

L_env = [None] *(L+1)
L_env[0-1] = np.array(1.,dtype=complex).reshape(1,1,1)


for i in range(0,5):
    mps, mpo, L_env, R_env = left_to_right_sweep(mps,mpo,L_env, R_env,einsum_eval)
    mps, mpo, L_env, R_env  = right_to_left_sweep(mps,mpo,L_env,R_env,einsum_eval)