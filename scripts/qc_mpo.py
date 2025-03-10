import numpy as np
from dmrg.initialization import single_site_operators
from itertools import product
from dmrg.fermions.mps import get_mps_from_occupation_numbers, get_random_mps
from dmrg.einsum_optimal_paths import EinsumEvaluator
from dmrg.heisenberg_chain.mps import create_neel_mps, right_canonicalize
from dmrg.heisenberg_chain.mpo import initialize_heisenberg_mpo
from dmrg.heisenberg_chain.sweep import precompute_right_environment,right_to_left_sweep, left_to_right_sweep
from dmrg.fermions.mpo import add_one_electron_interactions, reformat_mpo


import numpy as np

def create_local_mpo_tensors(one_el_integrals,N_sites,dim=4):
    
    total_N_ops = 2*N_sites **2

    mpo = np.zeros(shape = (N_sites, total_N_ops,dim,dim))

    # initialize all single-site operators to identity
    mpo[:] = np.eye(dim)

    one_e_indices = np.ndindex(one_el_integrals.shape)

    ## Spin up 1 e terms
    mpo = add_one_electron_interactions(mpo,one_e_indices,one_el_integrals,0,"up")

    #Spin down 1 e terms 
    mpo = add_one_electron_interactions(mpo,one_e_indices,one_el_integrals,N_sites**2,"down")

    return mpo




def embedd_operator(op, L, j):
    new_op = None
    for i in range(L):
        factor = op if i == j else np.eye(4)
        new_op = factor if new_op is None else np.kron(new_op, factor)
    return new_op

def construct_full_hamiltonian_operator(L, t_jk):

    d = 4**L

    H = np.zeros((d,d))

    index_set = np.ndindex(t_jk.shape)
    c_dag_up, c_up, c_dag_down, c_down = single_site_operators()


    #Spin up 
    for j,k in index_set:

        c_dag_up_j = embedd_operator(c_dag_up,L,j)
        c_up_k = embedd_operator(c_up,L,k)
        H += t_jk[j,k] * (c_dag_up_j @ c_up_k)

    for j,k in index_set:

        c_dag_down_j = embedd_operator(c_dag_down,L,j)
        c_down_k = embedd_operator(c_down,L,k)
        H += t_jk[j,k] * (c_dag_down_j @ c_down_k)

    return H





L = 4

t_matrix = np.random.normal(size = (L,L))
t_matrix = t_matrix + t_matrix.T

mpo = create_local_mpo_tensors(t_matrix, L)
mpo = reformat_mpo(mpo)


H_full = construct_full_hamiltonian_operator(L, t_matrix)
print("constructed Hfull")
eig_0, _ = np.linalg.eig(H_full)
print(np.sort(eig_0)[0])


einsum_eval = EinsumEvaluator(None)
mps = get_random_mps(L,1000)
mps = right_canonicalize(mps,einsum_eval)
R_env = precompute_right_environment(mps,mpo,einsum_eval)

for idx, env in enumerate(R_env):
    if env is None:
        print(f"{idx}: None")
    else:
        print(f"{idx}: {env.shape}")
        

L_env = [None] *(L+1)
L_env[0-1] = np.array(1.,dtype=float).reshape(1,1,1)


for i in range(0,5):
    mps, mpo, L_env, R_env = left_to_right_sweep(mps,mpo,L_env, R_env,einsum_eval)
    mps, mpo, L_env, R_env  = right_to_left_sweep(mps,mpo,L_env,R_env,einsum_eval)
