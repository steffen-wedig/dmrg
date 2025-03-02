import numpy as np

# Define basic 2x2 matrices for spin-1/2.
I = np.eye(2)
# Here, define your spin operators (ensure these are the correct representations)
Sx = np.array([[0, 0.5],
               [0.5, 0]])
Sy = np.array([[0, -0.5j],
               [0.5j, 0]])
Sz = np.array([[0.5, 0],
               [0, -0.5]])

def initialize_heisenberg_mpo(L, J):
    # MPO bond dimension is 5.
    D_mpo = 5
    mpo = []
    
    # First site: shape (1,5,2,2)
    W = np.zeros((1, D_mpo, 2, 2), dtype=complex)
    W[0, 0] = I      # (1,1)
    W[0, 1] = Sx     # (1,2)
    W[0, 2] = Sy     # (1,3)
    W[0, 3] = Sz     # (1,4)
    # W[0, 4] remains 0.
    mpo.append(W)
    
    # Intermediate sites: shape (5,5,2,2)
    for i in range(1, L-1):
        W = np.zeros((D_mpo, D_mpo, 2, 2), dtype=complex)
        W[0, 0] = I      # Propagate identity
        W[0, 1] = Sx     # "Emission" operators
        W[0, 2] = Sy
        W[0, 3] = Sz
        W[4, 4] = I      # Final propagation of identity
        W[1, 4] = J * Sx # Interaction terms
        W[2, 4] = J * Sy
        W[3, 4] = J * Sz
        mpo.append(W)
    
    # Last site: shape (5,1,2,2)
    W = np.zeros((D_mpo, 1, 2, 2), dtype=complex)
    W[0, 0] = I
    W[1, 0] = J * Sx
    W[2, 0] = J * Sy
    W[3, 0] = J * Sz
    W[4, 0] = I
    mpo.append(W)
    
    return mpo



def initialize_transverse_ising_mpo(L, J, h):

        d = 2

        sx = np.array([[0., 1.], [1., 0.]])
        sz = np.array([[1., 0.], [0., -1.]])
        id = np.eye(2)

        Ws = []
        # First vector Ws[0] = [1 , X , -g Z]
        w = np.zeros((1,3,d,d),dtype=complex)
        w[0,0] = id
        w[0,1] = sx
        w[0,2] = - h * sz
        Ws.append(w)
        
        # W[i] matrix
        w = np.zeros((3,3,d,d),dtype=complex)
        # top row: 1 X -g Z
        w[0,0] = id
        w[0,1] = sx
        w[0,2] = - h * sz
        # right column (-g Z , -J X ,1)^T
        #w[0,2] = - g * sz
        w[1,2] = - J * sx
        w[2,2] = id
        # create L-2 times the same matrix
        for i in range(1,L-1):
            Ws.append(w)
        # Last vector W[L-1]^T = [-g Z, -J X, 1]^T
        w = np.zeros((3,1,d,d),dtype=complex)
        w[0,0] = - h * sz
        w[1,0] = - J * sx
        w[2,0] = id
        Ws.append(w)
        return Ws