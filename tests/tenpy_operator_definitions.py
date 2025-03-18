import numpy as np
filling = 1

d = 4
states = ['empty', 'up', 'down', 'full']
# 0) Build the operators.
Nu_diag = np.array([0., 1., 0., 1.], dtype=np.float64)
Nd_diag = np.array([0., 0., 1., 1.], dtype=np.float64)
Nu = np.diag(Nu_diag)
Nd = np.diag(Nd_diag)
Ntot = np.diag(Nu_diag + Nd_diag)
dN = np.diag(Nu_diag + Nd_diag - filling)
NuNd = np.diag(Nu_diag * Nd_diag)
JWu = np.diag(1. - 2 * Nu_diag)  # (-1)^Nu
JWd = np.diag(1. - 2 * Nd_diag)  # (-1)^Nd
JW = JWu * JWd  # (-1)^{Nu+Nd
Cu = np.zeros((d, d))
Cu[0, 1] = Cu[2, 3] = 1
Cdu = np.transpose(Cu)
# For spin-down annihilation operator: include a Jordan-Wigner string JWu
# this ensures that Cdu.Cd = - Cd.Cdu
# c.f. the chapter on the Jordan-Wigner trafo in the userguide
Cd_noJW = np.zeros((d, d))
Cd_noJW[0, 2] = Cd_noJW[1, 3] = 1
Cd = np.dot(JWu, Cd_noJW)  # (don't do this for spin-up...)
Cdd = np.transpose(Cd)

print(JW)

print(f"Cu {Cu}")
print(f"Cdu {Cdu}")
print(f"Cd {Cd}")
print(f"Cdd {Cdd}")
