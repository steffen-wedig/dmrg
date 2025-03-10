import numpy as np
from dmrg.einsum_optimal_paths import EinsumEvaluator


def mps_norm(mps,einsum_eval):
        L = np.array([[1.0]])
        
        # Loop over each site in the MPS
        for A in mps:
            # A has shape (chi_left, d, chi_right)
            # Update the environment:
            # Here, we contract L (shape (chi_left, chi_left)) with A and its conjugate.
            # The contraction sums over the left bond of A and the physical index.
            L = einsum_eval('ab, asr, bsj->rj', L, A, A.conj())
            # Now L has shape (chi_right, chi_right)
        
        # At the end, L should be a 1x1 matrix; squeeze it to obtain a scalar.
        norm = L.squeeze()
        return norm

def tensor_to_mps(psi_coeff: np.ndarray):
    # psi_coeff: a tensor of shape (4, 4, ..., 4) with num_sites entries
    num_sites = psi_coeff.ndim
    mps = []
    current_tensor = psi_coeff

    # Left boundary dimension is 1 (start with a trivial bond)
    left_dim = 1

    for site in range(num_sites - 1):
        # Reshape current_tensor into a matrix:
        # Rows: combine left bond and the current physical index (dimension: left_dim * 4)
        # Columns: the rest of the indices (flattened)
        phys_dim = 4
        new_shape = (left_dim * phys_dim, -1)
        matrix = current_tensor.reshape(new_shape)

        # Perform SVD
        U, s, Vh = np.linalg.svd(matrix, full_matrices=False)


          # Truncate the SVD to the largest max_bond_dim singular values
        chi = min(50, len(s))
        U = U[:, :chi]
        s = s[:chi]
        Vh = Vh[:chi, :]



        # Determine the new bond dimension (chi)
        #chi = U.shape[1]

        # Reshape U into the MPS tensor for the current site with shape (left_dim, phys_dim, chi)
        A = U.reshape(left_dim, phys_dim, chi)
        mps.append(A)

        # Prepare the tensor for the next iteration:
        # Multiply s into Vh to form the new tensor.
        current_tensor = np.dot(np.diag(s), Vh)

        # Update the left bond dimension for the next iteration:
        left_dim = chi

    # The final tensor becomes the last site tensor, shape (left_dim, phys_dim)
    mps.append(current_tensor.reshape(left_dim, phys_dim, 1))
    return mps
    



def get_mps_from_occupation_numbers(occupation_numbers, bond_dimensions):

    d = 4 # dimension of the local Hilbert space
    mps = []
    L = occupation_numbers.shape[1]
    print(L)


    A = np.zeros(shape = (1,d,bond_dimensions))
    A[0,:,0] = occupation_numbers[:,0].T
    mps.append(A)

    for i in range(1,L-1):

            A = np.zeros(shape = (bond_dimensions,d,bond_dimensions))
            A[0,:,0] = occupation_numbers[:,i].T
            mps.append(A)

    
    A = np.zeros(shape = (bond_dimensions,d,1))
    A[0,:,0] = occupation_numbers[:,L-1].T
    mps.append(A)

    return mps

def get_random_mps(L,bond_dimensions):

    d = 4 # dimension of the local Hilbert space
    mps = []



    A = np.random.normal(loc = 0.0, scale = 1.0, size = (1,d,bond_dimensions))
    mps.append(A)

    for i in range(1,L-1):

            A = np.random.normal(loc = 0.0, scale = 1.0,size = (bond_dimensions,d,bond_dimensions))
            mps.append(A)

    
    A = np.random.normal(loc = 0.0, scale = 1.0,size = (bond_dimensions,d,1))
    mps.append(A)


    norm = mps_norm(mps,einsum_eval=EinsumEvaluator(None))

    mps[0] = mps[0] * 1/np.sqrt(norm)

    return mps



