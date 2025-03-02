import numpy as np 

from scipy.sparse.linalg import LinearOperator

from scipy.sparse import linalg



def effective_hamiltonian_action(psi, L_env, mpo, R_env, dims):


    # Reshape the input state psi into a tensor with the given dimensions.
    chi_left, d1, d2, chi_right = dims
    psi_tensor = psi.reshape(dims)
    
    # mpo bl-1 bl+1 sigmal sigmal+1 sigmal ' sigmal+1'
    # L = al-1 al-1' bl-1
    # P al-1' sigmal' sigma l+1' al+1'
    # R al+1' al+1 bl+1

    # al-1: i al-1': j  al+1: k  al+1' : l 
    # bl-1: m bl+1: n 
    # sigmal: o  sigmal+1: p  sigmal': q sigmal+1': r
    
    #print(R_env.shape) 
    #print(L_env.shape) 
    #print(mpo.shape) 
    #print(psi_tensor.shape) 


    result = np.einsum("mnopqr,ijm,jqrl,lkn->iopk",mpo,L_env,psi_tensor,R_env)
    return result.ravel()

def construct_effective_hamiltonian_operator(L_env, mpo, R_env, dims):
    """
    Construct a LinearOperator representing the effective Hamiltonian.
    
    Parameters:


      L_env, mpo1, mpo2, R_env: tensors as described above.
      dims : tuple (chi_left, d1, d2, chi_right) describing the state space.
      
    Returns:
      A scipy.sparse.linalg.LinearOperator that implements the effective Hamiltonian.
    """
    # Total dimension of the two-site state.

    N = np.prod(dims)
    
   
    def matvec(psi):
        return effective_hamiltonian_action(psi, L_env, mpo, R_env, dims)
    
    # Create and return the LinearOperator.
    return LinearOperator((N, N), matvec=matvec, dtype=complex)



