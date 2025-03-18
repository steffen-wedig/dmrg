import numpy as np

from dmrg.spin_systems.mps import create_neel_mps 
from dmrg.spin_systems.mpo import initialize_heisenberg_mpo
from dmrg.dmrg.sweep import precompute_right_environment, right_to_left_sweep, left_to_right_sweep
from dmrg.dmrg.mps import right_canonicalize


from dmrg.einsum_evaluation import EinsumEvaluator
# Initialize an MPS for a chain of L sites
L = 20  # for example
D= 5

J = 1.0
mps = create_neel_mps(L,D)

mpo = initialize_heisenberg_mpo(L,J)
R_env = [None] * (L+1)

einsum_eval = EinsumEvaluator()

mps = right_canonicalize(mps, einsum_eval)

R_env = precompute_right_environment(mps,mpo,einsum_eval)

for idx, env in enumerate(R_env):
    if env is None:
        print(f"{idx}: None")
    else:
        print(f"{idx}: {env.shape}")
        

L_env = [None] *(L+1)
L_env[0-1] = np.array(1.,dtype=complex).reshape(1,1,1)


for i in range(0,5):
    mps, mpo, L_env, R_env = left_to_right_sweep(mps,mpo,L_env, R_env,einsum_eval)
    mps, mpo, L_env, R_env  = right_to_left_sweep(mps,mpo,L_env,R_env,einsum_eval)

