import numpy as np


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