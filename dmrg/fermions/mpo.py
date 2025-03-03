import numpy as np
from dmrg.initialization import single_site_operators, get_operators_for_spin
from numpy.linalg import matrix_power

def create_local_mpo_tensors(one_el_integrals,two_el_integrals,N_sites,dim=4):
    
    total_N_ops = 2*N_sites **2 + 4* N_sites**4

    mpo = np.zeros(shape = (N_sites, total_N_ops,dim,dim))

    # initialize all single-site operators to identity
    mpo[:] = np.eye(dim)

    one_e_indices = np.ndindex(one_el_integrals.shape)

    ## Spin up 1 e terms
    add_one_electron_interactions(mpo,one_e_indices,one_el_integrals,0,"up")

    #Spin down 1 e terms 
    add_one_electron_interactions(mpo,one_e_indices,one_el_integrals,N_sites**2,"down")

    two_e_indices = np.ndindex(two_el_integrals.shape)

    add_two_eletron_interactions(mpo,two_e_indices,two_el_integrals,start = 2*(N_sites**2),sigma_spin="up",tau_spin="up")

    add_two_eletron_interactions(mpo,two_e_indices,two_el_integrals,start = 2*(N_sites**2)+N_sites**4,sigma_spin="up",tau_spin="down")

    add_two_eletron_interactions(mpo,two_e_indices,two_el_integrals,start = 2*(N_sites**2)+2*(N_sites**4),sigma_spin="down",tau_spin="up")

    add_two_eletron_interactions(mpo,two_e_indices,two_el_integrals,start = 2*(N_sites**2)+3*(N_sites**4),sigma_spin="down",tau_spin="down")


    return mpo


  


def add_one_electron_interactions(mpo,index_array,one_electron_integrals, start,sigma_spin):

    c_dag_sigma, c_sigma = get_operators_for_spin(sigma_spin)

    F = np.diag([1,-1,-1,1])

    for interaction_counter,indices in enumerate(index_array,start):
        
        # Multiply the value into the first site operator
        mpo[0,interaction_counter,:,:] = mpo[0,interaction_counter,:,:]*one_electron_integrals[indices]

        creation_op_site_index = indices[0]

        mpo[creation_op_site_index,interaction_counter,:,:] = mpo[creation_op_site_index,interaction_counter,:,:] @ matrix_power(F,creation_op_site_index) @ c_dag_sigma

        annihilation_op_site_index = indices[1]

        mpo[annihilation_op_site_index,interaction_counter,:,:] = mpo[annihilation_op_site_index,interaction_counter,:,:] @ matrix_power(F,annihilation_op_site_index) @c_sigma



def add_two_eletron_interactions(mpo,index_array,two_electron_integrals, start,sigma_spin,tau_spin):
    
    c_dag_sigma, c_sigma = get_operators_for_spin(sigma_spin)

    c_dag_tau, c_tau = get_operators_for_spin(tau_spin)
    F = np.diag([1,-1,-1,1])


    for interaction_counter, indices in enumerate(index_array,start = start):
        
        # Multiply the value into the first site operator
        mpo[0,interaction_counter,:,:] = mpo[0,interaction_counter,:,:]*0.5*two_electron_integrals[indices]
        
        creation_op_site_index_0 = indices[0]

        mpo[creation_op_site_index_0,interaction_counter,:,:] = mpo[creation_op_site_index_0,interaction_counter,:,:] @  matrix_power(F,creation_op_site_index_0) @ c_dag_sigma

        creation_op_site_index_1 = indices[1]

        mpo[creation_op_site_index_1,interaction_counter,:,:] = mpo[creation_op_site_index_1,interaction_counter,:,:] @ matrix_power(F,creation_op_site_index_1) @ c_dag_tau

        annihilation_op_site_index_0 = indices[2]

        mpo[annihilation_op_site_index_0,interaction_counter,:,:] = mpo[annihilation_op_site_index_0,interaction_counter,:,:] @ matrix_power(F,annihilation_op_site_index_0) @ c_tau

        annihilation_op_site_index_1 = indices[3]

        mpo[annihilation_op_site_index_1,interaction_counter,:,:] = mpo[annihilation_op_site_index_1,interaction_counter,:,:] @ matrix_power(F,annihilation_op_site_index_1) @ c_sigma

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