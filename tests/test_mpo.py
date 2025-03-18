from pyscf import gto, scf, ao2mo
import numpy as np
from pyscf import fci
from dmrg.einsum_evaluation import EinsumEvaluator

from dmrg.fermions.op_construction import construct_molecular_mpo, convert_mpo_to_sparse

from dmrg.fermions.mps import tensor_to_mps
from dmrg.fermions.mpo import create_local_mpo_tensors, reformat_mpo_sparse, contract_expectation

from dmrg.fermions.mps import mps_norm



def test_fermionic_mpo():
    N_hydrogen = 2
    nuclear_distance = 1

    H_chain_string = "; ".join([f"H 0 0 {nuclear_distance*i}" for i in range(N_hydrogen)])
    
    mol = gto.M(
        atom = H_chain_string,  # Adjust bond length if necessary
        basis = '3-21G',
        symmetry = False
    )
    mf = scf.RHF(mol)
    mf.kernel()  # Run HF calculation
    N_orbitals = len(mf.mo_occ)
    h1e = mf.get_hcore()  # One-electron integrals
    h2e = ao2mo.kernel(mol, mf.mo_coeff,aosym = "s1").reshape((N_orbitals,N_orbitals, N_orbitals, N_orbitals)) 



    fci_h2 = fci.FCI(mf)
    e_fci, ci_coeff = fci_h2.kernel()

    print(e_fci)


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

    mps = tensor_to_mps(psi_coeff,D=50)

    mpo = construct_molecular_mpo(h1e,h2e)

    mpo = convert_mpo_to_sparse(mpo)


    einsum_eval = EinsumEvaluator("sparse")
    
    
    mps_n = mps_norm(mps,einsum_eval)

    mps[0] = mps[0] * 1/np.sqrt(mps_n)

    mps_n2 = mps_norm(mps,einsum_eval)
    
    print(mps_n2)


    E = contract_expectation(mps,mpo, einsum_eval)
    print(E)


    


test_fermionic_mpo()