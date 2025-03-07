from pyscf import gto, scf, ao2mo
import numpy as np
from pyscf import fci
from dmrg.einsum_optimal_paths import EinsumEvaluator


from dmrg.fermions.mps import tensor_to_mps
from dmrg.fermions.mpo import create_local_mpo_tensors, reformat_mpo_sparse, contract_expectation



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

def test_fermionic_mpo():
    N_hydrogen = 2
    nuclear_distance = 1

    H_chain_string = "; ".join([f"H 0 0 {nuclear_distance*i}" for i in range(N_hydrogen)])
    
    mol = gto.M(
        atom = H_chain_string,  # Adjust bond length if necessary
        basis = 'cc-pVDZ',
        symmetry = False
    )
    mf = scf.RHF(mol)
    mf.kernel()  # Run HF calculation
    N_orbitals = len(mf.mo_occ)
    h1e = mf.get_hcore()  # One-electron integrals
    h2e = ao2mo.kernel(mol, mf.mo_coeff,aosym = "s1").reshape((N_orbitals,N_orbitals, N_orbitals, N_orbitals)) 



    fci_h2 = fci.FCI(mf)
    e_fci, ci_coeff = fci_h2.kernel()

    print(ci_coeff)

    psi_coeff = np.zeros(shape = (4,)*N_orbitals)

    for coef_indices in np.ndindex((N_orbitals,N_orbitals)):
        index_vec = np.zeros(shape = psi_coeff.ndim,dtype=int)
        
        if coef_indices[0] == coef_indices[1]:
            index_vec[coef_indices[0]] = 3

        else:
            index_vec[coef_indices[0]] = 1
            index_vec[coef_indices[1]] = 2

        psi_coeff[index_vec] = ci_coeff[coef_indices]

    mps = tensor_to_mps(psi_coeff)

#   


    #print(h1e.shape)
    mpo = create_local_mpo_tensors(h1e,h2e,N_sites=N_orbitals)

    #Reformat the MPO
    mpo = reformat_mpo_sparse(mpo)


    for mpo_loc in mpo:
        print(mpo_loc.shape)

    einsum_eval = EinsumEvaluator("sparse")
    
    
    mps_n = mps_norm(mps,einsum_eval)

    mps[0] = mps[0] * 1/np.sqrt(mps_n)

    mps_n2 = mps_norm(mps,einsum_eval)
    print(mps_n)
    print(mps_n2)


    E = contract_expectation(mps,mpo, einsum_eval)
    print(E)


    


test_fermionic_mpo()