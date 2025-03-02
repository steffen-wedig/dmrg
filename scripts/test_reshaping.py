import numpy as np
from scipy.sparse import cs

o_e = np.ones(shape=(10,10))

t_w = np.ones(shape = (10,10,10,10))
idx = np.ndindex(t_w.shape)
N_sites = 10 


a = np.zeros(shape = (N_sites, N_sites**4,4,4))

a[:] = np.eye(4)

for no, indices in enumerate(idx):
    print(indices)
    for idx in indices:
        print(idx)
        a[idx,no,:,:] = np.eye(4)


for i in range(len(a)):
    print(i)
    A = np.zeros((N_sites**4, N_sites**4, 4, 4))
    A[np.arange(N_sites**4), np.arange(N_sites**4)] = a[0,:,:,:]