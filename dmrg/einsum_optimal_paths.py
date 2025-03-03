import opt_einsum as oe



def get_optimal_paths(einsum_string, *operators):
    shapes = tuple(op.shape for op in operators)
    expr = oe.contract_expression(einsum_string,*(op.shape for op in operators))

    return {f"{shapes}": expr}



class EinsumEvaluator():


    #TODO