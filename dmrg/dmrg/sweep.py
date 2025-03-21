import numpy as np
from dmrg.dmrg.effective_hamiltonian import (
    construct_effective_hamiltonian_operator,
)
from scipy.sparse.linalg import eigsh
from dmrg.einsum_evaluation import EinsumEvaluator


def update_right_environment(mps_i, mpo_i, R_env_i_next, einsum_eval: EinsumEvaluator):
    """
    R_env i+1: al, al', bl
    M[i] al-1' sigmal' al'
    W bl-1 bl sigmal sigmal'
    M[i]* al-1  sigmal al
    al : i, al' : j, bl : k
    al-1: l, al-1': m , bl-1:n
    sigmal o: , sigmal': p
    """

    R_env_i = einsum_eval(
        "mpj,nkop,ijk,loi->mln", mps_i, mpo_i, R_env_i_next, mps_i.conj()
    )

    return R_env_i


def update_left_environment(mps_i, mpo_i, L_env_prev, einsum_eval: EinsumEvaluator):
    """
    - a_i : i
    - b_i : j
    - a_i' : k
    - s_i : l
    - s_i' : m
    - a_{i-1} : n
    - b_{i-1} : o
    - a_{i-1}' : p
    """

    L_env = einsum_eval("ojlm,nli,npo,pmk->ikj", mpo_i, mps_i.conj(), L_env_prev, mps_i)

    return L_env


def precompute_right_environment(mps, mpo, einsum_eval: EinsumEvaluator):
    L = len(mps)

    R_env = [None] * (L + 1)
    R_env[L] = np.array(1.0).reshape((1, 1, 1))

    for i in range(L - 1, 1, -1):
        R_env[i] = update_right_environment(mps[i], mpo[i], R_env[i + 1], einsum_eval)
    return R_env


def combine_sites(A_0, A_1, einsum_eval):

    return einsum_eval("ijk,klm->ijlm", A_0, A_1)


def right_to_left_sweep(mps, mpo, L_env, R_env, einsum_eval):
    L = len(mps)

    evs = []
    for i in range(L - 1, 0, -1):
        M = combine_sites(mps[i - 1], mps[i], einsum_eval)

        dims = M.shape
        W = einsum_eval("ijkl,jmno->imknlo", mpo[i - 1], mpo[i])

        h_eff_op = construct_effective_hamiltonian_operator(
            L_env[i - 2], W, R_env[i + 1], dims, einsum_eval
        )

        psi_0 = M.ravel()

        eigenvalue, eigenvector = eigsh(h_eff_op, k=1, which="SA", v0=psi_0)
        evs.append(eigenvalue)
        M_updated = eigenvector.reshape((dims[0] * dims[1], dims[2] * dims[3]))
        U, S, Vh = np.linalg.svd(M_updated, full_matrices=False)

        D_left, d1, d2, D_right = dims

        D_max = 5

        r = min(D_max, U.shape[1])
        U = U[:, :r]

        S = S[:r]

        Vh = Vh[:r, :]
        # Reshape U into the updated tensor for site i.
        new_tensor_left = (U @ np.diag(S)).reshape(D_left, d1, r)

        # Absorb S into Vh to form the updated tensor for site i+1.
        new_tensor_right = Vh.reshape(r, d2, D_right)

        mps[i - 1] = new_tensor_left
        mps[i] = new_tensor_right

        R_env[i] = update_right_environment(mps[i], mpo[i], R_env[i + 1], einsum_eval)

    return mps, mpo, L_env, R_env, evs


def left_to_right_sweep(mps, mpo, L_env, R_env, einsum_eval: EinsumEvaluator):

    L = len(mps)
    Evs = []
    for i in range(0, L - 1):
        # print(f"Step {i}")
        M = combine_sites(mps[i], mps[i + 1], einsum_eval)
        dims = M.shape

        # Two site MPO
        W = einsum_eval("ijkl,jmno->imknlo", mpo[i], mpo[i + 1])
        h_eff_op = construct_effective_hamiltonian_operator(
            L_env[i - 1], W, R_env[i + 2], dims, einsum_eval
        )

        psi_0 = M.ravel()

        eigenvalue, eigenvector = eigsh(h_eff_op, k=1, which="SA", v0=psi_0)
        Evs.append(eigenvalue)
        M_updated = eigenvector.reshape((dims[0] * dims[1], dims[2] * dims[3]))
        U, S, Vh = np.linalg.svd(M_updated, full_matrices=False)

        D_left, d1, d2, D_right = dims

        D_max = 5

        r = min(D_max, U.shape[1])
        U = U[:, :r]

        S = S[:r]

        Vh = Vh[:r, :]
        # Reshape U into the updated tensor for site i.
        new_tensor_left = U.reshape(D_left, d1, r)

        # Absorb S into Vh to form the updated tensor for site i+1.
        new_tensor_right = (np.diag(S) @ Vh).reshape(r, d2, D_right)

        mps[i] = new_tensor_left
        mps[i + 1] = new_tensor_right

        L_env[i] = update_left_environment(
            mps_i=mps[i], mpo_i=mpo[i], L_env_prev=L_env[i - 1], einsum_eval=einsum_eval
        )

    return mps, mpo, L_env, R_env, Evs
