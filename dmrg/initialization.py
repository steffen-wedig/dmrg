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
    c_dag_up = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    c_up = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
    ])

    c_dag_down = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
    ])
    c_down = np.array([
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, -1.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])

    return c_dag_up, c_up, c_dag_down, c_down 

def construct_initial_one_state_hamiltonian(h1e, h2e):
    c_dag_up, c_up, c_dag_down, c_down = single_site_operators()

    n_up = np.dot(c_dag_up, c_up)
    n_down = np.dot(c_dag_down, c_down)

    H = h1e * (n_up+n_down) + h2e * np.dot(n_up, n_down)

