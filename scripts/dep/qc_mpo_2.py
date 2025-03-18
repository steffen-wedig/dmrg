from dmrg.fermions.mpo import create_local_mpo_tensors, reformat_mpo, reformat_mpo_sparse
from pyscf import gto, scf, ao2mo
import numpy as np
from dmrg.utils import single_site_operators
from dmrg.fermions.mps import get_mps_from_occupation_numbers, get_random_mps, mps_norm
from dmrg.einsum_evaluation import EinsumEvaluator

from dmrg.spin_systems.mps import create_neel_mps, right_canonicalize
from dmrg.spin_systems.mpo import initialize_heisenberg_mpo
from dmrg.dmrg.sweep import precompute_right_environment, right_to_left_sweep, left_to_right_sweep


N_orbitals = 4

h1e = np.random.normal(size = (N_orbitals,N_orbitals))
h2e = np.random.normal(size = (N_orbitals,N_orbitals, N_orbitals, N_orbitals))


mpo = create_local_mpo_tensors(h1e,h2e,N_sites=N_orbitals)

#Reformat the MPO
mpo = reformat_mpo_sparse(mpo)

#remove the zero elementsö

from dmrg.utils import get_initial_states_from_mol_orb_occ
mps = get_random_mps(N_orbitals,500)


einsum_eval = EinsumEvaluator()
print(mps_norm(mps,einsum_eval))

L = len(mps)

R_env = [None] * (L+1)



mps = right_canonicalize(mps, einsum_eval)

R_env = precompute_right_environment(mps,mpo,einsum_eval)
       

L_env = [None] *(L+1)
L_env[0-1] = np.array(1.).reshape(1,1,1)


for i in range(0,5):
    mps, mpo, L_env, R_env = left_to_right_sweep(mps,mpo,L_env, R_env,einsum_eval)
    mps, mpo, L_env, R_env  = right_to_left_sweep(mps,mpo,L_env,R_env,einsum_eval)