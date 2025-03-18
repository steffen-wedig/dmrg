from dataclasses import dataclass
from typing import Sequence
from dmrg.utils import get_operator_for_character_and_spin
import numpy as np 


@dataclass
class Operator:
    site: int
    character: int  # 1 for the creation op, 0 for the annihilation op
    spin: int  # 0 for up, 1 for down


def group_operators(ops: Sequence[Operator]):
    characters = tuple(op.character for op in ops)
    spins = tuple(op.spin for op in ops)
    sites = tuple(op.site for op in ops)
    return characters, spins, sites

class OperatorChunk():
    def __init__(self, operators: Sequence[Operator]):
        self.operators = operators
        assert len(set([op.site for op in operators])) == 1

        self.site = operators[-1].site

    def construct_local_operator(self):

        local_op = np.eye(4)

        for op in self.operators:
            A = get_operator_for_character_and_spin(op.character, op.spin)
            local_op = local_op @ A
        self.matrix_op = local_op
        return local_op

    def premultiply_by_prefactor(self,A):
        self.matrix_op = A @ self.matrix_op