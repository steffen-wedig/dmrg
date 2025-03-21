import numpy as np
from dmrg.einsum_evaluation import EinsumEvaluator
from dmrg.spin_systems.mps import create_neel_mps
from dmrg.spin_systems.mpo import initialize_transverse_ising_mpo
from dmrg.dmrg.sweep import (
    precompute_right_environment,
    right_to_left_sweep,
    left_to_right_sweep,
)
from dmrg.dmrg.mps import right_canonicalize
from dmrg.spin_systems.exact import exact_E
from dmrg.utils import contract_expectation
from dmrg.plotting import plot_energy_minimization_in_sweep
import time

# Initialize an MPS for a chain of L sites
L = 20
D = 10  # With Bond dimension D

N_sweeps = 3

J = 1.0  # Strength of the nearest neighbour coupling
h = 0.5  # Coupling with the magnetic field

mps = create_neel_mps(L, D)
einsum_eval = EinsumEvaluator()
mps = right_canonicalize(mps, einsum_eval)


# Initializing MPO, left and right environment
mpo = initialize_transverse_ising_mpo(L, J, h)

R_env = [None] * (L + 1)
R_env = precompute_right_environment(mps, mpo, einsum_eval)

L_env = [None] * (L + 1)
L_env[0 - 1] = np.array(1.0).reshape(1, 1, 1)

energies = []  # Storing the lowest eigenvalues during the sweep


start_time = time.time()
for i in range(0, N_sweeps):
    mps, mpo, L_env, R_env, energies_left = left_to_right_sweep(
        mps, mpo, L_env, R_env, einsum_eval
    )
    mps, mpo, L_env, R_env, energies_right = right_to_left_sweep(
        mps, mpo, L_env, R_env, einsum_eval
    )
    energies.extend(energies_left)
    energies.extend(energies_right)
end_time = time.time()

print(f"Finished DMRG in {end_time-start_time} seconds")

E = contract_expectation(mps, mpo, einsum_eval)
print(f"Energy MPS-MPO contraction {E}")
reference_energy = exact_E(L, J, h)
print(f"Reference Energy {reference_energy}")

print(f"Energy above reference {E-reference_energy}")
fig = plot_energy_minimization_in_sweep(energies, reference_energy)
fig.savefig("Transverse_Ising_Energy_Minimization.pdf", bbox_inches="tight")
