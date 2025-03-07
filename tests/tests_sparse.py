import sparse as sp


import numpy as np

a = np.eye(3)

b = sp.COO.from_numpy(a)

c = a@b

print(type(c))