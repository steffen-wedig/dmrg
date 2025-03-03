import numpy as np



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

    return mps