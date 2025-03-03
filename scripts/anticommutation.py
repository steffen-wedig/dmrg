from dmrg.initialization import single_site_operators
import numpy as np
from numpy.linalg import matrix_power

c_dag_up, c_dag_down, c_up, d_down = single_site_operators()



F = np.diag([1.,-1.,-1.,1.])
print(F)

a = matrix_power(F,1) @ c_dag_up

print(c_dag_up)

print(a)