import numpy as np
import sparse
from dmrg.fermions.operator_wrangling import Operator
from dmrg.fermions.op_indexing import (
    OperatorTree,
    get_spin_num,
    collect_numpy_arrays,
)
from dmrg.fermions.finite_state_machine import FiniteStateMachine
from dmrg.utils import (
    get_jordan_wigner_transformation_matrix,
)
from typing import Sequence
from itertools import pairwise


def initialize_empty_mpo(N_sites,N_op_states,d_dim):
    
    mpo = []
    for  _ in range(0,N_sites):
        W = np.zeros((N_op_states,N_op_states,d_dim,d_dim))
        mpo.append(W)
    return mpo



def add_jordan_wigner_matrices(mpo, op_tree: OperatorTree):

    JW = np.expand_dims(get_jordan_wigner_transformation_matrix(4), 0)
    ID = np.expand_dims(np.eye(4), 0)
    # Unpack all indices in the op tree
    single_indices = collect_numpy_arrays(op_tree.singles)
    doubles_indices = collect_numpy_arrays(op_tree.doubles)
    triples_indices = collect_numpy_arrays(op_tree.triples)

    A = np.tile(JW, reps=(len(single_indices), 1, 1))
    B = np.tile(ID, reps=(len(doubles_indices), 1, 1))
    C = np.tile(JW, reps=(len(triples_indices), 1, 1))

    diagonal_indices = np.concat((single_indices, doubles_indices, triples_indices))
    diagonal = np.concat((A, B, C), axis=0)

    for i in range(1, len(mpo) - 1):
        mpo[i][diagonal_indices, diagonal_indices, :, :] = diagonal


def construct_molecular_mpo(h1e, h2e):

    N_sites = h1e.shape[0]
    d_dim = 4  # Dimension of the local Fock Space
    op_tree = OperatorTree(N_sites) # Operator tree is used to retrieve the correct indices for each operator. ( Indices in the context of the finite state machine )
    N_op_states = op_tree.get_total_N_ops()
    # Initialize empty mpo

    mpo = initialize_empty_mpo(N_sites, N_op_states, d_dim)
    add_jordan_wigner_matrices(
        mpo, op_tree
    )  # add jordan wigner matrices on the diagonal

    fsm = FiniteStateMachine(op_tree)

    # Here, we add all possible combinations of spins into the mpo matrix
    add_one_electron_integrals(mpo, h1e, "up", fsm)
    add_one_electron_integrals(mpo, h1e, "down", fsm)

    add_two_electron_integrals(mpo, h2e, "up", "up", fsm)
    add_two_electron_integrals(mpo, h2e, "up", "down", fsm)
    add_two_electron_integrals(mpo, h2e, "down", "up", fsm)
    add_two_electron_integrals(mpo, h2e, "down", "down", fsm)

    #Remove the array padding in the first and last mpo matrices, because these should be vectors
    shape_except_last = list(mpo[0].shape)
    shape_except_last[0] -= 1
    assert np.allclose(mpo[0][:-1,:,:,:],np.zeros(shape=shape_except_last))
    mpo[0] = np.expand_dims(mpo[0][-1,:,:,:],0)

    shape_except_last = list(mpo[-1].shape)
    shape_except_last[1] -= 1
    assert np.allclose(mpo[-1][:,1:,:,:],np.zeros(shape=shape_except_last))
    mpo[-1] = np.expand_dims(mpo[-1][:,0,:,:],1)


    return mpo


def sort_operators(operators: Sequence[Operator]):

    N = len(operators)
    indices = np.array([op.site for op in operators])
    sorting_indices = np.argsort(indices)
    permutation_mat = np.zeros(shape=(N, N), dtype=int)
    for i, j in zip(sorting_indices, np.arange(N)):
        permutation_mat[i, j] = 1

    parity = np.linalg.det(permutation_mat)

    sorted_operators = [operators[idx] for idx in sorting_indices]

    return sorted_operators, parity





def add_operator_string_to_mpo(
    mpo, operators: Sequence[Operator], fsm: FiniteStateMachine, prefactor: float
):

    N_sites = len(mpo)
    d_dim = mpo[0].shape[-1]
    JW = get_jordan_wigner_transformation_matrix(d_dim=d_dim)

    sorted_operators, parity = sort_operators(operators)

    state_transition, operator_chunks = fsm.get_state_transition(sorted_operators)
    
    # Get all matrix representations
    for ops in operator_chunks:
        ops.construct_local_operator()

    operator_chunks[0].premultiply_by_prefactor(parity * prefactor * JW)


    for (state_0, state_1), ops in zip(
        pairwise(state_transition), reversed(operator_chunks)
    ):
    
        site = ops.site

        if state_1 == fsm.op_tree.N_ops-1:
            mpo[site][state_1, state_0, :, :] = mpo[site][state_1, state_0, :, :] + ops.matrix_op

        else: 

            if np.allclose(mpo[site][state_1, state_0, :, :], np.zeros((d_dim, d_dim))):
                mpo[site][state_1, state_0, :, :] = ops.matrix_op
            else:
                assert np.allclose(mpo[site][state_1, state_0, :, :], ops.matrix_op)

       
    

def add_one_electron_integrals(
    mpo, h1e, spin, finite_state_machine: FiniteStateMachine
):

    spin_int = get_spin_num(spin)
    one_el_indices = np.ndindex(h1e.shape)

    for i, j in one_el_indices:

        creation_op = Operator(site=i, character=1, spin=spin_int)
        annihilation_op = Operator(site=j, character=0, spin=spin_int)

        # Figure out all the state transitions
        operators = (creation_op, annihilation_op)

        prefactor = h1e[i, j]

        add_operator_string_to_mpo(mpo, operators, finite_state_machine, prefactor)


def add_two_electron_integrals(
    mpo, h2e, spin_sigma, spin_tau, finite_state_machine: FiniteStateMachine
):

    spin_sigma_int = get_spin_num(spin_sigma)
    spin_tau_int = get_spin_num(spin_tau)

    two_el_indices = np.ndindex(h2e.shape)

    for p, q, r, s in two_el_indices:

        creation_op_p = Operator(site=p, character=1, spin=spin_sigma_int)
        creation_op_q = Operator(site=q, character=1, spin=spin_tau_int)
        annihilation_op_r = Operator(site=r, character=0, spin=spin_tau_int)
        annihilation_op_s = Operator(site=s, character=0, spin=spin_sigma_int)

        # Figure out all the state transitions
        operators = (creation_op_p, creation_op_q, annihilation_op_r, annihilation_op_s)

        prefactor = 0.5 * h2e[p, q, r, s]

        add_operator_string_to_mpo(mpo, operators, finite_state_machine, prefactor)


def local_op_matrix_to_sparse(A):
    coords = np.nonzero(A)
    data = A[coords]
    # Build and return the sparse COO tensor
    return sparse.COO(coords, data, shape=A.shape)


def convert_mpo_to_sparse(mpo):
    sparse_mpo = []
    for A in mpo:
        sparse_mpo.append(local_op_matrix_to_sparse(A))

    return sparse_mpo

