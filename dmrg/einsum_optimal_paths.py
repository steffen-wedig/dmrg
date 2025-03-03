import opt_einsum as oe
from typing import Callable



class EinsumEvaluator():

    def __init__(self):

        self.einsum_contractions = {}

    def __call__(self, einsum_str : str, *operators):
        shapes = tuple(op.shape for op in operators)

        if einsum_str in self.einsum_contractions and shapes in self.einsum_contractions[einsum_str]:
            expr : Callable = self.einsum_contractions[einsum_str][shapes] 
        else:
            expr = self.get_optimal_paths(einsum_str,shapes)
            self.einsum_contractions[einsum_str] = {shapes: expr}

        return expr(*operators)
        



    @staticmethod
    def get_optimal_paths(einsum_string, shapes):
        return oe.contract_expression(einsum_string,*shapes)
