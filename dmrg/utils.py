import numpy as np
from dmrg.einsum_evaluation import EinsumEvaluator


def get_state_from_occ(occ):

    match occ:
        case 0:
            state = np.array([1.0, 0.0, 0.0, 0.0])  # Empty Molecular orbital
        case 1:
            state = np.array([0.0, 1.0, 0.0, 0.0])  # Spin up orbital
        # case 0: state = np.array([1.,0.,0.,0.]) # TODO: Do we care about the spin up vs spin down difference?
        case 2:
            state = np.array(
                [0.0, 0.0, 0.0, 1.0]
            )  # Corresponds to spin up, spin down occupation

    return state.reshape(-1, 1)


def get_initial_states_from_mol_orb_occ(mo_occ):
    """
    Gets the initial state vectors for all HF molecular orbitals.
    """

    occs = [get_state_from_occ(occ) for occ in mo_occ]
    return np.hstack(occs)


def single_site_operators():
    # Spin up creation operator
    c_dag_up = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],   
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    c_up = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    c_dag_down = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
        ]
    )
    c_down = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    return c_dag_up, c_up, c_dag_down, c_down


def get_operator_for_character_and_spin(character,spin):

    c_dag_up, c_up, c_dag_down, c_down = single_site_operators()

    if character == 1 and spin == 0:
        return c_dag_up
    elif character == 1 and spin == 1:
        return c_dag_down
    elif character == 0 and spin == 0:
        return c_up
    elif character == 0 and spin == 1 :
        return c_down
    else: 
        raise ValueError()
    

def get_operators_for_spin(spin):

    c_dag_up, c_up, c_dag_down, c_down = single_site_operators()

    if spin == "up":
        return c_dag_up, c_up

    if spin == "down":
        return c_dag_down, c_down


def get_jordan_wigner_transformation_matrix(d_dim):

    if d_dim == 2:
        return np.array([[1,0],[0,-1]])
    elif d_dim == 4:
        return np.array([[ 1.,  0.,  0.,  0.], [ 0., -1.,  0.,  0.],  [ 0. , 0. , -1. , 0.],  [ 0. , 0. , 0. , 1.]])
    else:
        raise ValueError("JW transformation only implemented for Fock space dimensions 2 and 4 ")
    


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