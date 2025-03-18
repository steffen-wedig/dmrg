import numpy as np
from dmrg.einsum_evaluation import EinsumEvaluator




def create_neel_mps(L, D):
    """
    Create an MPS for a Neel state on an L-site spin-1/2 chain with bond dimension D.
    
    Parameters:
        L (int): Number of sites.
        D (int): Desired bond dimension.
        
    Returns:
        list of numpy.ndarray: The MPS tensors.
            - First tensor has shape (1, 2, D)
            - Intermediate tensors have shape (D, 2, D)
            - Last tensor has shape (D, 2, 1)
    """
    mps = []
    # Local physical dimension for spin-1/2
    d = 2
    
    for i in range(L):
        # Determine the spin at site i: even -> up, odd -> down.
        if i % 2 == 0:
            # Spin up: vector [1, 0]
            spin_vector = np.array([1.0, 0.0])
        else:
            # Spin down: vector [0, 1]
            spin_vector = np.array([0.0, 1.0])
        
        if i == 0:
            # First site: shape (1, d, D)
            A = np.zeros((1, d, D))
            # Place the nonzero element in the first "bond" index.
            A[0, :, 0] = spin_vector
        elif i == L - 1:
            # Last site: shape (D, d, 1)
            A = np.zeros((D, d, 1))
            A[0, :, 0] = spin_vector
        else:
            # Intermediate sites: shape (D, d, D)
            A = np.zeros((D, d, D))
            A[0, :, 0] = spin_vector
            A[1:, : , 1:] = np.random.normal(0,1, size = (D-1,2,D-1),)

        mps.append(A)
    
    return mps

