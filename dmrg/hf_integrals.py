from pyscf import ao2mo, gto, scf, fci
import numpy as np
from dmrg.dmrg.mps import tensor_to_mps

def get_H2_integrals():
    mol = gto.M(
        atom="H 0 0 0; H 0 0 1.1",  # Adjust bond length if necessary
        basis="3-21G",
    )

    return get_pyscf_itegrals(mol)


def get_H2_MO_integrals():
    mol = gto.M(
        atom="H 0 0 0; H 0 0 1.1",  # Adjust bond length if necessary
        basis="3-21G",
    )
    return get_pyscf_integrals_mo(mol)


def get_N2_integrals(mol):
    mol = gto.M(
        atom="N 0 0 0; N 0 0 1.1",  # Adjust bond length if necessary
        basis="cc-pVDZ",
        symmetry=True,
    )
    return get_pyscf_itegrals(mol)


def get_pyscf_itegrals(mol):
    mf = scf.RHF(mol)
    mf.kernel()  # Run HF calculation
    mo_coeff = mf.mo_coeff
    N_orbitals = len(mf.mo_occ)
    h1e = mf.get_hcore()
    h2e = mol.intor('int2e', aosym='s1').reshape(
        (N_orbitals, N_orbitals, N_orbitals, N_orbitals)
    )
    
    #ao2mo.kernel(mol, mo_coeff, aosym="s1").reshape(
    #    (N_orbitals, N_orbitals, N_orbitals, N_orbitals)
    #)
    return h1e, h2e

def get_pyscf_integrals_mo(mol):

    mf = scf.RHF(mol)
    mf.kernel()

    # Get the one-electron core Hamiltonian in the AO basis
    hcore_ao = mf.get_hcore()
    N_orbitals = len(mf.mo_occ)
    # Get the molecular orbital coefficients
    mo_coeff = mf.mo_coeff

    # Transform the one-electron integrals to the MO basis
    h1e = np.dot(mo_coeff.T, np.dot(hcore_ao, mo_coeff))

    h2e = ao2mo.kernel(mol, mo_coeff, aosym="s1").reshape((N_orbitals, N_orbitals, N_orbitals, N_orbitals))

    return h1e, h2e


def get_fci_mps(D):
    mol = gto.M(
        atom="H 0 0 0; H 0 0 1.1",  # Adjust bond length if necessary
        basis="3-21G",
    )
    mf = scf.RHF(mol)
    mf.kernel()  # Run HF calculation
    N_orbitals = len(mf.mo_occ)


    fci_h2 = fci.FCI(mf)
    e_fci, ci_coeff = fci_h2.kernel()

    psi_coeff = np.zeros(shape = (4,)*N_orbitals)

    for coef_indices in np.ndindex((N_orbitals,N_orbitals)):
        index_vec = np.zeros(shape = psi_coeff.ndim,dtype=int)
        
        if coef_indices[0] == coef_indices[1]:
            index_vec[coef_indices[0]] = 3

        else:
            index_vec[coef_indices[0]] = 1
            index_vec[coef_indices[1]] = 2

        psi_coeff[index_vec] = ci_coeff[coef_indices]

    mps = tensor_to_mps(psi_coeff,D)

    return e_fci, mps