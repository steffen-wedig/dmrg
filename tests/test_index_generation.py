from dmrg.fermions.op_indexing import OperatorTree, IndexGenerator
import numpy as np
from dmrg.fermions.op_construction import sort_operators, construct_local_operator, add_jordan_wigner_matrices
from dmrg.fermions.op_indexing import Operator, collect_numpy_arrays
from dmrg.fermions.finite_state_machine import FiniteStateMachine
from itertools import pairwise
from dmrg.fermions.mpo import initialize_empty_mpo

N_sites = 4

h1e = np.random.normal(size=(N_sites,N_sites))
h2e = np.random.normal(size=(N_sites,N_sites,N_sites,N_sites))

idx_generator = IndexGenerator()
op_tree = OperatorTree(N_sites, idx_generator)

N_sites = h1e.shape[0]
d_dim = 4  # Dimension of the local Fock Space
op_tree = OperatorTree(N_sites)
N_op_states = op_tree.get_total_N_ops()
print(N_op_states)


mpo = initialize_empty_mpo(N_sites, N_op_states, d_dim)
mpo = add_jordan_wigner_matrices(mpo,op_tree) 



fsm = FiniteStateMachine(op_tree)

spin_int = 0

creation_op_0 = Operator(site = 2, character = 1, spin=0)
creation_op_1 = Operator(site = 1, character = 1, spin=1)
#annihilation_op_2 = Operator(site = 1, character = 0, spin=spin_int)
#annihilation_op_3 = Operator(site = 1, character = 0, spin=spin_int)


operators = (creation_op_0,creation_op_1)#,annihilation_op_2, annihilation_op_3)
sorted_operators, parity = sort_operators(operators)


state_transition, operator_chunks = fsm.get_state_transition(operators=sorted_operators)
for (state_0, state_1), ops in zip(pairwise(state_transition),reversed(operator_chunks)):
            
    site = ops[-1].site
    local_fock_op = construct_local_operator(ops)
            #print(local_fock_op)