from dmrg.fermions.operator_wrangling import Operator, group_operators, OperatorChunk

from dmrg.fermions.op_indexing import OperatorTree
from typing import Sequence



class FiniteStateMachine:

    def __init__(self, op_tree: OperatorTree):

        self.op_tree = op_tree

    def get_state_transition(self, operators: Sequence[Operator]) -> tuple[list[int], Sequence[OperatorChunk]]:

        # Break down the sequence of operators into chunks, where each chunck should have operators that act on the same site
        
        op_chunks = self.get_op_chunks(operators)
        state_transition_list = [0]
        seen_operators = []

        # Reverse through the chunks to get the indixes of the state transitions
        for op_chunk in reversed(op_chunks[1:]):
            seen_operators.extend(op_chunk.operators)

            characters, spins, sites = group_operators(seen_operators)
            state_index = self.op_tree.get_index(characters, spins, sites).item()
            state_transition_list.append(state_index)

        state_transition_list.append(self.op_tree.N_ops-1)
        return state_transition_list, op_chunks

    @staticmethod
    def get_op_chunks(operators: Sequence[Operator]) -> Sequence[OperatorChunk]:


        # This function separates an operator string into chunks. This is necesary because two operators in a string can be on the same site, so they have to be placed into the same local mpo matrix.
        op_chunks = []

        running_chunk = [operators[0]]

        for op in operators[1:]:

            if op.site == running_chunk[-1].site:
                running_chunk.append(op)
            else:
                op_chunks.append(OperatorChunk(running_chunk))
                running_chunk = [op]

        op_chunks.append(OperatorChunk(running_chunk))

        return op_chunks
