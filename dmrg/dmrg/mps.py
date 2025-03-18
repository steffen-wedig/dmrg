import numpy as np
from dmrg.einsum_evaluation import EinsumEvaluator



def right_canonicalize(mps, einsum_eval: EinsumEvaluator):

    L = len(mps)
    
    for i in range(L-1,0,-1):

        D_left, d, D_right= mps[i].shape
        A = mps[i].reshape((D_left, (d * D_right)))
        U,S,VH = np.linalg.svd(A,full_matrices= False)
        

        r = VH.shape[0]
        VH = VH.reshape(r,d,D_right)
        mps[i] = VH

        A_prev = mps[i-1]

        US = U @ np.diag(S)

        A_new = einsum_eval("ijk,kl->ijl",A_prev,US)
        mps[i-1]=A_new

    return mps