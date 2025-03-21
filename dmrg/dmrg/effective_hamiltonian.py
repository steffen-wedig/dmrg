import numpy as np 

from scipy.sparse.linalg import LinearOperator
from dmrg.einsum_evaluation import EinsumEvaluator


def effective_hamiltonian_action(psi, L_env, mpo, R_env, dims,einsum_eval):


    # Reshape the input state psi into a tensor with the given dimensions.
    psi_tensor = psi.reshape(dims)
    

    # Tensors with their algebraic dimensions
    # mpo bl-1 bl+1 sigmal sigmal+1 sigmal ' sigmal+1'
    # L = al-1 al-1' bl-1
    # P al-1' sigmal' sigma l+1' al+1'
    # R al+1' al+1 bl+1
    # Algebraic indices to einstein indices
    # al-1: i al-1': j  al+1: k  al+1' : l 
    # bl-1: m bl+1: n 
    # sigmal: o  sigmal+1: p  sigmal': q sigmal+1': r
    
    result = einsum_eval("mnopqr,ijm,jqrl,lkn->iopk",mpo,L_env,psi_tensor,R_env)
    return result.ravel()

def construct_effective_hamiltonian_operator(L_env, mpo, R_env, dims, einsum_eval: EinsumEvaluator):
    # Total dimension of the two-site state.

    N = np.prod(dims)
    
   
    def matvec(psi):
        return effective_hamiltonian_action(psi, L_env, mpo, R_env, dims,einsum_eval)
    
    # Create and return the LinearOperator.
    return LinearOperator((N, N), matvec=matvec)



