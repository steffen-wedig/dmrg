import numpy as np
from dmrg.utils import single_site_operators, get_operators_for_spin
from numpy.linalg import matrix_power
import sparse
from dmrg.einsum_evaluation import EinsumEvaluator

def create_local_mpo_tensors(one_el_integrals,two_el_integrals,N_sites,dim=4):
    
    total_N_ops = 2*N_sites **2 #+ 4* N_sites**4

    mpo = np.zeros(shape = (N_sites, total_N_ops,dim,dim))

    # initialize all single-site operators to identity
    mpo[:] = np.eye(dim)

    one_e_indices = np.ndindex(one_el_integrals.shape)

    ## Spin up 1 e terms
    mpo = add_one_electron_interactions(mpo,one_e_indices,one_el_integrals,0,"up")

    #Spin down 1 e terms 
    mpo = add_one_electron_interactions(mpo,one_e_indices,one_el_integrals,N_sites**2,"down")

    #two_e_indices = np.ndindex(two_el_integrals.shape)
#
    #mpo = add_two_eletron_interactions(mpo,two_e_indices,two_el_integrals,start = 2*(N_sites**2),sigma_spin="up",tau_spin="up")
#
    #mpo = add_two_eletron_interactions(mpo,two_e_indices,two_el_integrals,start = 2*(N_sites**2)+N_sites**4,sigma_spin="up",tau_spin="down")
#
    #mpo = add_two_eletron_interactions(mpo,two_e_indices,two_el_integrals,start = 2*(N_sites**2)+2*(N_sites**4),sigma_spin="down",tau_spin="up")
#
    #mpo = add_two_eletron_interactions(mpo,two_e_indices,two_el_integrals,start = 2*(N_sites**2)+3*(N_sites**4),sigma_spin="down",tau_spin="down")
#

    return mpo


  


def add_one_electron_interactions(mpo,index_array,one_electron_integrals, start,sigma_spin):

    c_dag_sigma, c_sigma = get_operators_for_spin(sigma_spin)

    F = np.diag([1,-1,-1,1])

    for interaction_counter,indices in enumerate(index_array,start):
        
        # Multiply the value into the first site operator
        mpo[0,interaction_counter,:,:] = mpo[0,interaction_counter,:,:]*one_electron_integrals[indices]

        creation_op_site_index = indices[0]

        mpo[creation_op_site_index,interaction_counter,:,:] = mpo[creation_op_site_index,interaction_counter,:,:] @ c_dag_sigma

        annihilation_op_site_index = indices[1]

        for i in range(min(creation_op_site_index,annihilation_op_site_index), max(creation_op_site_index, annihilation_op_site_index)):
            mpo[i,interaction_counter,:,:] = mpo[i,interaction_counter,:,:] @ F


        mpo[annihilation_op_site_index,interaction_counter,:,:] = mpo[annihilation_op_site_index,interaction_counter,:,:]  @c_sigma

    return mpo


def add_two_eletron_interactions(mpo,index_array,two_electron_integrals, start,sigma_spin,tau_spin):
    
    c_dag_sigma, c_sigma = get_operators_for_spin(sigma_spin)

    c_dag_tau, c_tau = get_operators_for_spin(tau_spin)
    F = np.diag([1,-1,-1,1])


    for interaction_counter, indices in enumerate(index_array,start = start):
        
        # Multiply the value into the first site operator
        mpo[0,interaction_counter,:,:] = mpo[0,interaction_counter,:,:]*0.5*two_electron_integrals[indices]
        
        creation_op_site_index_0 = indices[0]

        mpo[creation_op_site_index_0,interaction_counter,:,:] = mpo[creation_op_site_index_0,interaction_counter,:,:] @ c_dag_sigma

        creation_op_site_index_1 = indices[1]

        mpo[creation_op_site_index_1,interaction_counter,:,:] = mpo[creation_op_site_index_1,interaction_counter,:,:] @ c_dag_tau

        annihilation_op_site_index_0 = indices[2]

        mpo[annihilation_op_site_index_0,interaction_counter,:,:] = mpo[annihilation_op_site_index_0,interaction_counter,:,:] @ c_tau

        annihilation_op_site_index_1 = indices[3]

        mpo[annihilation_op_site_index_1,interaction_counter,:,:] = mpo[annihilation_op_site_index_1,interaction_counter,:,:] @ c_sigma

        

    return mpo

def reformat_mpo(mpo):
    L = mpo.shape[0]
    num_ops = mpo.shape[1]

    list_mpo = []

    A0 = np.zeros((1,num_ops,4,4))
    A0[0, np.arange(num_ops)] = mpo[0,:,:,:]
    list_mpo.append(A0)

    for i in range(1,L-1):
        A = np.zeros((num_ops, num_ops, 4, 4))
        A[np.arange(num_ops), np.arange(num_ops)] = mpo[i,:,:,:]
        list_mpo.append(A)

    AL = np.zeros((num_ops,1,4,4))
    AL[np.arange(num_ops),0] = mpo[L-1,:,:,:]

    list_mpo.append(AL)

    return list_mpo

def reformat_mpo_sparse(mpo):

    L, num_ops, d1, d2 = mpo.shape 
    list_mpo = []

    # --- First Block: A0 ---
    shape0 = (1, num_ops, d1, d2)
    coords_list = []
    data_list = []
    # Only nonzero entries come from mpo[0, :, :, :]
    for k in range(num_ops):
        for r in range(d1):
            for c in range(d2):
                # Coordinate: (0, k, r, c)
                coords_list.append((0, k, r, c))
                data_list.append(mpo[0, k, r, c])
    coords_arr = np.array(coords_list).T  # Transpose to shape (4, num_entries)
    A0 = sparse.COO(coords_arr, data_list, shape=shape0)
    list_mpo.append(A0)

    # --- Intermediate Blocks ---
    # For each intermediate MPO tensor: shape (num_ops, num_ops, 4, 4)
    # Only the diagonal blocks are nonzero: for each k, at position (k, k, :, :)
    for i in range(1, L-1):
        shape_i = (num_ops, num_ops, d1, d2)
        coords_list = []
        data_list = []
        for k in range(num_ops):
            for r in range(d1):
                for c in range(d2):
                    # Only fill the diagonal block at (k, k, :, :)
                    coords_list.append((k, k, r, c))
                    data_list.append(mpo[i, k, r, c])
        coords_arr = np.array(coords_list).T
        A = sparse.COO(coords_arr, data_list, shape=shape_i)
        list_mpo.append(A)

    # --- Last Block: AL ---
    # Shape: (num_ops, 1, 4, 4)
    shapeL = (num_ops, 1, d1, d2)
    coords_list = []
    data_list = []
    for k in range(num_ops):
        for r in range(d1):
            for c in range(d2):
                # Coordinate: (k, 0, r, c)
                coords_list.append((k, 0, r, c))
                data_list.append(mpo[L-1, k, r, c])
    coords_arr = np.array(coords_list).T
    AL = sparse.COO(coords_arr, data_list, shape=shapeL)
    list_mpo.append(AL)

    return list_mpo




def contract_expectation(mps, mpo,einsum_eval : EinsumEvaluator):
    # Initialize left environment as a scalar wrapped in a tensor of shape (1, 1, 1)
    L = np.array([[[1.0]]]) # shape (1, 1, 1)
    
    # Loop over each site
    for A, W in zip(mps, mpo):
        # A has shape (chi_left, d, chi_right)
        # W has shape (w_left, w_right, d, d)
        # L has shape (chi_left, chi_left, w_left)
        #
        # We want to update L to a new tensor L_new with shape (chi_right, chi_right, w_right)
        #
        # Contraction (using explicit index labels):
        #   L[a, b, p] · A[a, s, i] · A*[b, s', j] · W[p, q, s, s']
        #
        # The resulting tensor L_new[i, j, q] is computed as:
        L = einsum_eval("ojlm,nli,npo,pmk->ikj", W, A.conj(),L,A)
        # Now, L has shape (chi_right, chi_right, w_right)
    
    # At the end, L should have shape (1, 1, 1) (i.e., a scalar wrapped in a tensor)
    energy = L.squeeze() 
    return energy




def initialize_empty_mpo(N_sites,N_op_states,d_dim):
    
    mpo = []
    for  _ in range(0,N_sites):
        W = np.zeros((N_op_states,N_op_states,d_dim,d_dim))
        mpo.append(W)
    return mpo