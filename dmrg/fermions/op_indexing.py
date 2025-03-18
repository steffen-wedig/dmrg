import numpy as np

def get_spin_num(spin :str):

    if spin == "up":
        return 0
    elif spin == "down":
        return 1
    else:
        raise ValueError


class IndexGenerator:

    def __init__(self, start=1):

        self.start = start
        self.count = start

    def get_next_index_array(self, shape: tuple):

        N_elements = np.prod(shape)

        indices = np.arange(
            start=self.count, stop=self.count + N_elements, dtype=int
        ).reshape(shape)

        self.count = self.count + N_elements

        return indices


class OperatorTree:

    def __init__(self, N_sites, index_generator: IndexGenerator = IndexGenerator()):
        self.N_sites = N_sites

        self.index_generator = index_generator
        self.singles = self.get_singles()
        self.doubles = self.get_doubles()
        self.triples = self.get_triples()
        self.N_ops = self.get_total_N_ops()

    def get_singles(self):

        singles = {}

        ops = [(0,), (1,)]
        spins = [(0,), (1,)]

        for op in ops:
            singles[op] = {
                spin: self.index_generator.get_next_index_array(shape=(self.N_sites))
                for spin in spins
            }

        return singles

    def get_doubles(self):

        doubles = {}

        ops = [(0, 0), (0, 1), (1, 0), (1, 1)]
        spins = [(0, 0), (0, 1), (1, 0), (1, 1)]

        for op in ops:
            doubles[op] = {
                spin: self.index_generator.get_next_index_array(
                    shape=(self.N_sites, self.N_sites)
                )
                for spin in spins
            }

        return doubles

    def get_triples(self):

        triples = {}

        ops = [(1,1,1),(1, 1, 0), (1, 0, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1),(0,0,0)]
        spins = [(1,1,1),(1, 1, 0), (1, 0, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1),(0,0,0)]

        for op in ops:
            triples[op] = {
                spin: self.index_generator.get_next_index_array(
                    shape=(self.N_sites, self.N_sites, self.N_sites)
                )
                for spin in spins
            }

        return triples

    def get_index(self, characters, spins, sites):

        match len(characters):
            case 3:
                return self.triples[characters][spins][sites]
            case 2:
                return self.doubles[characters][spins][sites]
            case 1:
                return self.singles[characters][spins][sites]
            case 4:
                print("Here")
                return self.N_ops

    def get_total_N_ops(self):
        max_count = self.index_generator.count.item() +1# The way the count is implemented (starting at 1 and moving ahead by 1 after the new indices have been added to the tree), it already accounts for the initial state. Final state has to be added

        return max_count


def get_index_sorting(indices: tuple[int,...]):

    N = len(indices)
    indices = np.array(indices)
    print(indices)
    sorting_indices = np.argsort(np.array(indices))
    print(sorting_indices)

    permutation_mat = np.zeros(shape = (N,N),dtype=int)

    for i,j in zip(sorting_indices,np.arange(N)):

        print(i,j)
        permutation_mat[i,j] = 1

    print(permutation_mat)

    parity = np.linalg.det(permutation_mat)

    return indices[sorting_indices], sorting_indices, parity


def collect_numpy_arrays(nested):
    arrays = []
    if isinstance(nested, dict):
        for key, value in nested.items():
            arrays.extend(collect_numpy_arrays(value))
    elif isinstance(nested, np.ndarray):
        arrays.append(nested)

    arrays = np.concatenate([arr.flatten() for arr in arrays])
    return arrays