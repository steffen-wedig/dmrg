import numpy as np
from dmrg.hf_integrals import get_H2_integrals, get_fci_mps
from dmrg.fermions.op_construction import construct_molecular_mpo, convert_mpo_to_sparse
from dmrg.einsum_evaluation import EinsumEvaluator
from dmrg.dmrg.sweep import (
    precompute_right_environment,
    right_to_left_sweep,
    left_to_right_sweep,
)
from dmrg.dmrg.mps import mps_norm
from dmrg.utils import contract_expectation
from dmrg.plotting import plot_energy_minimization_in_sweep


# Disclaimer: This script does not return the correct energies. The correct construction of the local matrices of the MPO are


L = 4  # Chain length
d_dim = 4  # Dimension of the local hilbert space
D = 10  # Bond dimension

h1e, h2e = get_H2_integrals()

mpo = construct_molecular_mpo(h1e, h2e)

mpo = convert_mpo_to_sparse(mpo)


einsum_eval = EinsumEvaluator()
R_env = [None] * (L + 1)

e_fci, mps = get_fci_mps(D)
norm = mps_norm(mps, einsum_eval)
mps[0] = mps[0] * 1 / np.sqrt(norm)  # MPS normalization

R_env = precompute_right_environment(mps, mpo, einsum_eval)

L_env = [None] * (L + 1)
L_env[0 - 1] = np.array(
    1.0,
).reshape(1, 1, 1)

energies = []
for i in range(0, 5):
    mps, mpo, L_env, R_env, Evs_left = left_to_right_sweep(
        mps, mpo, L_env, R_env, einsum_eval
    )
    mps, mpo, L_env, R_env, Evs_right = right_to_left_sweep(
        mps, mpo, L_env, R_env, einsum_eval
    )

    energies.extend(Evs_left)
    energies.extend(Evs_right)


print(f"Norm of the Matrix Product State after DMRG {mps_norm(mps,einsum_eval)}")
E = contract_expectation(mps, mpo, einsum_eval)
print(f"Contracting the MPO and MPS yields a energy of {E}")


fig = plot_energy_minimization_in_sweep(energies, e_fci)
fig.savefig("Transverse_Ising_Energy_Minimization.pdf", bbox_inches="tight")
