import numpy as np
from dmrg.utils import single_site_operators
from itertools import product
from dmrg.fermions.mps import get_mps_from_occupation_numbers, get_random_mps
from dmrg.einsum_evaluation import EinsumEvaluator
from dmrg.spin_systems.mps import create_neel_mps, right_canonicalize
from dmrg.spin_systems.mpo import initialize_heisenberg_mpo
from dmrg.dmrg.sweep import precompute_right_environment,right_to_left_sweep, left_to_right_sweep
from dmrg.fermions.mpo import add_one_electron_interactions, reformat_mpo
from dmrg.fermions.mpo import create_local_mpo_tensors, reformat_mpo, reformat_mpo_sparse
from pyscf import gto, scf, ao2mo
import numpy as np
from dmrg.utils import single_site_operators
from dmrg.fermions.mps import get_mps_from_occupation_numbers, get_random_mps, mps_norm
from dmrg.einsum_evaluation import EinsumEvaluator

from dmrg.spin_systems.mps import create_neel_mps, right_canonicalize
from dmrg.spin_systems.mpo import initialize_heisenberg_mpo
from dmrg.dmrg.sweep import precompute_right_environment, right_to_left_sweep, left_to_right_sweep

import numpy as np

def create_local_mpo_tensors_spin_chain(t_pq,N_sites):
    creation_op = np.array([[0 ,0],[1,0]])
    annihilation_op = np.array([[0 ,1],[0,0]])

    
    # 2 spins * 2 ops * N_sites + initial + final state
    mpo_bond_order = 2*(2*N_sites)+ 2


    dim = 2
    id = np.eye(dim)
    
    mpo = []
    jw_block = np.array([[1,0],[0,-1]])
    upper_tr = np.triu(t_pq,k=1)

    t_slice = upper_tr[0,:].reshape(1,N_sites, 1, 1)
    W0 = np.zeros((1,mpo_bond_order,dim,dim))
    W0[0,0,:,:] = 2* t_pq[0,0] * creation_op @ annihilation_op
    W0[0,1:N_sites+1,:,:] = t_slice * jw_block@creation_op
    W0[0,N_sites+1:2*N_sites+1,:,:] = - t_slice * jw_block@annihilation_op
    W0[0,2*N_sites+1:3*N_sites+1,:,:] = t_slice * jw_block@creation_op
    W0[0,3*N_sites+1:4*N_sites+1,:,:] = - t_slice * jw_block@annihilation_op
    
    W0[0,-1,:,:] = id

    print(W0)

    mpo.append(W0)

    for site in range(1,N_sites-1):
        
        t_slice = upper_tr[site,:].reshape(1,N_sites, 1, 1)
        W = np.zeros((mpo_bond_order,mpo_bond_order,dim,dim))
        W[0,0,:,:] = id
        
        # Add the self-interaction term
        W[-1,0,:,:] = 2* t_pq[site,site] * (creation_op @ annihilation_op)

        # add the dagger operators for termination
        W[-1,1:N_sites+1,:,:] = t_slice * jw_block@ creation_op

        # Adds the annihilation ops for termination 
        W[-1,N_sites+1:2*N_sites+1,:,:] = - t_slice * jw_block@ annihilation_op

        W[-1,2*N_sites+1:3*N_sites+1,:,:] = t_slice * jw_block@creation_op
        W[-1,3*N_sites+1:4*N_sites+1,:,:] = - t_slice * jw_block@annihilation_op

        #lower right corner identity
        W[-1,-1,:,:] = id

        # Add the initiation operators
        W[1+site,0,:,:] = annihilation_op
        W[N_sites+site+1,0,:,:] = creation_op
        W[1+site+2*N_sites,0,:,:] = annihilation_op
        W[1+site+3*N_sites,0,:,:] = creation_op

        W[1:N_sites+1,1:N_sites+1,:,:] = create_jordan_wigner_block_diagonal(site,N_sites)

        W[N_sites+1:2*N_sites+1,N_sites+1:2*N_sites+1,:,:] = create_jordan_wigner_block_diagonal(site,N_sites)

        W[2*N_sites+1:3*N_sites+1,2*N_sites+1:3*N_sites+1,:,:] = create_jordan_wigner_block_diagonal(site,N_sites)

        W[3*N_sites+1:4*N_sites+1,3*N_sites+1:4*N_sites+1,:,:] = create_jordan_wigner_block_diagonal(site,N_sites)



        mpo.append(W)

        
        # Add the JW ops

    WK = np.zeros((mpo_bond_order,1,dim,dim))
    WK[0,0,:,:] = id
    WK[-1,0,:,:] = 2* t_pq[-1,-1] * creation_op @ annihilation_op
    WK[N_sites,0,:,:] = annihilation_op
    WK[2*N_sites,0,:,:] = creation_op
    WK[3*N_sites,0,:,:] = annihilation_op
    WK[4*N_sites,0,:,:] = creation_op
    mpo.append(WK)

    return mpo


def create_jordan_wigner_block_diagonal(k,N_sites):


    # Create a block tensor of zeros with shape (N, N, 2, 2)
    jw = np.zeros((N_sites, N_sites,  2, 2), dtype=float)
    
    jw_block = np.array([[1,0],[0,-1]])
    # Fill diagonal blocks from index k to the last block with matrix A
    for i in range(k+1,N_sites):
        print(i)
        jw[i, i] = jw_block
        
    return jw


def embedd_operator(op, L, j):
    new_op = None
    for i in range(L):
        factor = op if i == j else np.eye(2)
        new_op = factor if new_op is None else np.kron(new_op, factor)
    return new_op

def construct_full_hamiltonian_operator(L, t_jk):

    d = 2**(2*L)

    print(d)

    H = np.zeros((d,d))

    index_set = np.ndindex(t_jk.shape)
    creation_op = np.array([[0 ,0],[1,0]])
    annihilation_op = np.array([[0 ,1],[0,0]])


    #Spin up 
    for j,k in index_set:

        c_dag_up_j = embedd_operator(creation_op,2*L,j)
        c_up_k = embedd_operator(annihilation_op,2*L,k)
        H += t_jk[j,k] * (c_dag_up_j @ c_up_k)

    for j,k in index_set:
        c_dag_down_j = embedd_operator(creation_op,2*L,j + 0.5*(2**(2*L)))
        c_down_k = embedd_operator(annihilation_op,2*L,k + 0.5*(2**(2*L)))
        H += t_jk[j,k] * (c_dag_down_j @ c_down_k)

    return H



N_sites = 4
mol = gto.M(
    atom = 'H 0 0 0; H 0 0 1.1',  # Adjust bond length if necessary
    basis = '3-21G',
    symmetry = True
)

mf = scf.RHF(mol)
mf.kernel()  # Run HF calculation
mo_coeff = mf.mo_coeff 
N_orbitals = len(mf.mo_occ)
h1e = mf.get_hcore()


print(h1e)
print(h1e.shape)

from pyscf import fci
fci_h2 = fci.FCI(mf)
e_fci = fci_h2.kernel()[0]
print(f"FCI {e_fci}")





mpo = create_local_mpo_tensors_spin_chain(h1e,N_sites)

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)
for i, op in enumerate(mpo):
    N_ops = (4*N_sites+2)*2
    if i == 0:
        pp_op = op.reshape(N_ops,2)
        
    elif i == N_sites-1:
        pp_op = op.reshape(N_ops,2)
        
    else:

        A = op.transpose(0, 2, 1, 3)
        pp_op= A.reshape(N_ops, N_ops)

  
    print(pp_op)
    print("\n\n\n")

mps = get_random_mps(N_sites,100,2)


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



H_full = construct_full_hamiltonian_operator(L= N_sites,t_jk=h1e)
print("constructed Hfull")
eig_0, _ = np.linalg.eig(H_full)
print(np.sort(eig_0)[0])