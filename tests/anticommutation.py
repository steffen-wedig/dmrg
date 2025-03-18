from dmrg.utils import single_site_operators
import numpy as np

c_dag_up, c_up, c_dag_down, c_down = single_site_operators()


spin_up_state = np.array([0,1,0,0]).reshape(-1,1)

spin_down_state = np.array([0,0,1,0]).reshape(-1,1)

print(np.dot(c_dag_up, spin_down_state))
print(c_dag_up@c_dag_up)