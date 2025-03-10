import numpy as np

from dmrg.heisenberg_chain.mps import create_neel_mps, right_canonicalize
from dmrg.heisenberg_chain.mpo import initialize_heisenberg_mpo
from dmrg.heisenberg_chain.sweep import precompute_right_environment, right_to_left_sweep, left_to_right_sweep

from dmrg.einsum_optimal_paths import EinsumEvaluator
# Initialize an MPS for a chain of L sites
L = 20  # for example
D= 5

J = 1.0
mps = create_neel_mps(L,D)

mpo = initialize_heisenberg_mpo(L,J)
R_env = [None] * (L+1)

einsum_eval = EinsumEvaluator(None)

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

