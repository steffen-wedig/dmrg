import numpy as np
from dmrg.utils import single_site_operators
from dmrg.spin_systems.mps import create_neel_mps, right_canonicalize
import numpy as np
import sparse
from dmrg.fermions.mpo import reformat_mpo, reformat_mpo_sparse, create_local_mpo_tensors

def test_anticommutation_relations_single_site_operators():
    c_dag_up, c_up, c_dag_down, c_down = single_site_operators()

    print(c_up @ c_dag_down + c_dag_down @ c_up)

    assert np.all(( c_dag_up @ c_dag_down + c_dag_down @ c_dag_up) == 0.)
    assert np.all(( c_up @ c_down + c_down @ c_up) == 0.)
    assert np.array_equal((c_up @ c_dag_up + c_dag_up @ c_up),np.eye(4))
    assert np.array_equal((c_down @ c_dag_down + c_dag_down @ c_down),np.eye(4))


def test_right_canoicalization():
    L = 10
    D= 5

    D_max = 10
    J = 1.0
    mps = create_neel_mps(L,D)
    mps = right_canonicalize(mps)

    for A in mps:
        
        contraction = np.einsum("ijk,ljk->il",A,A.conj())
        assert np.all(contraction == np.eye(contraction.shape[0]))



def test_sparse_mpo():


    mpo = create_local_mpo_tensors(np.random.normal(size=(2,2)), np.random.normal(size = (2,2,2,2)),2,4)

    mpo_list_sparse = reformat_mpo_sparse(mpo)

    mpo_list = reformat_mpo(mpo)

    for sparse, dense in zip(mpo_list_sparse, mpo_list):
        assert np.allclose(sparse.todense(),dense)



test_sparse_mpo()