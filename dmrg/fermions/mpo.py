import numpy as np
from dmrg.initialization import single_site_operators


def create_local_mpo_tensors(one_el_integrals,two_el_integrals):

    mpo = []

    N_sites = one_el_integrals.shape[0]

    for site_idx in range(N_sites):
        
        one_el_ops = get_one_electron_ops(one_el_integrals,site_idx)


        two_el_ops = get_two_electron_ops(two_el_integrals, site_idx)

        local_ops = [*one_el_ops, *two_el_ops]
        
        mpo.append(local_ops)

    return mpo


def get_one_electron_ops(one_el_integrals, site_idx):

    one_el_integrals_vec = one_el_integrals.reshape(
        -1,
    )

    N_sites = one_el_integrals.shape[0]
    N_1e_per_spin = N_sites**2
    N_1e = 2 * N_1e_per_spin
    c_dag_up, c_up, c_dag_down, c_down = single_site_operators()

    local_op = get_identity_local_op(N_1e)

    if site_idx == 0:
        for i in range(N_1e_per_spin):
            local_op[i] *= one_el_integrals_vec[i]
        for i in range(N_1e_per_spin):
            local_op[i + N_1e_per_spin] *= one_el_integrals_vec[i]

    creation_op_indices = list(range(site_idx * N_sites, N_sites * (site_idx + 1)))

    annihilation_op_indices = list(range(site_idx, N_1e_per_spin, N_sites))
    for creation_op_idx in creation_op_indices:
        local_op[creation_op_idx] = local_op[creation_op_idx] @ c_dag_up
    for creation_op_idx in creation_op_indices:
        local_op[creation_op_idx + N_1e_per_spin] = (
            local_op[creation_op_idx + N_1e_per_spin] @ c_dag_down
        )
    for annihilation_op_idx in annihilation_op_indices:
        local_op[annihilation_op_idx] = local_op[annihilation_op_idx] @ c_up
    for annihilation_op_idx in annihilation_op_indices:
        local_op[annihilation_op_idx + N_1e_per_spin] = (
            local_op[annihilation_op_idx + N_1e_per_spin] @ c_down
        )

    return local_op


def get_two_electron_ops(two_electron_integral, site_idx):


    two_electron_integral_vec = two_electron_integral.reshape(-1,)

    N_sites = two_electron_integral.shape[0]
    N_2e_per_spin = N_sites **4
    N_2e = 2* N_2e_per_spin


    assert (two_electron_integral.ndim == 2)


    local_op = get_identity_local_op(N_ops=N_2e)

    if site_idx == 0:
        for i in range(N_2e_per_spin):
            local_op[i] *= two_electron_integral_vec[i]
        for i in range(N_2e_per_spin):
            local_op[i + N_2e_per_spin] *= two_electron_integral_vec[i]
            

    creation_ops = list(range(site_idx * N_sites, N_sites * (site_idx + 1)))+ list(range(site_idx * N_sites, N_sites * (site_idx + 1)))




    return local_op

def get_identity_local_op(N_ops):
    I = np.eye(4)
    local_op = [I] * N_ops
    return local_op
